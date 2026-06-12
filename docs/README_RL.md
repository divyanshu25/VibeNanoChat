# Reinforcement Learning, from First Principles

*A primer on how nanochat does RL. Assumes you've understood pretraining and SFT and nothing past that.*

> **Note on file paths.** VibeNanoChat itself does not (yet) ship an RL stage — it covers pretraining and SFT-time evals. The RL code dissected below lives in the **upstream [nanochat](https://github.com/karpathy/nanochat) repo**, so every `path/to/file.py` reference here is relative to *that* repo, not this one.

---

## Where we are, and why SFT isn't enough

By the time you reach RL you've done two things to the model:

1. **Pretraining** — predict the next token over a giant pile of internet text. The model learns language, facts, a sloppy world model.
2. **SFT (supervised fine-tuning)** — show the model a few thousand curated `(conversation, ideal_response)` pairs and again do next-token prediction, but now only on the assistant's tokens. The model learns the *format* of being a helpful assistant: turn-taking, tool calls, answering instead of continuing.

Both of these are the same hammer: **imitation**. You show the model a target string and say "make this string more likely." That's cross-entropy loss, the whole way down.

Here's the problem. Imitation has a ceiling, and the ceiling is the data. With SFT you can only ever teach the model to copy the answers you already wrote down. But for a task like grade-school math (GSM8K), what we actually want is not "memorize this one solution path." We want the model to **reason its way to the correct final number**, and there are a thousand valid reasoning paths to any given answer. We don't have demonstrations for all of them. More importantly, the thing we actually care about is a property of the *outcome* — "is the final number correct?" — not a property of any particular string.

SFT can't optimize that directly. SFT optimizes "match this string." We want to optimize "get the answer right, by whatever path." Those are different objectives, and the gap between them is exactly the gap that RL fills.

```
SFT:   here is the correct answer, make it more likely.   (imitation)
RL:    try to answer. I'll tell you if you were right.    (trial and error)
       now make the things that worked more likely.
```

That's the whole pitch. Let's build it up properly.

---

## The setup: turning a chat model into an RL agent

We need to recast "answer a math question" as something RL can chew on. The vocabulary of RL is: **policy**, **action**, **trajectory**, **reward**.

- The **policy** is the model itself. Given a context, it outputs a probability distribution over the next token. Call it $\pi_\theta(\text{token} \mid \text{context})$ where $\theta$ are the model's weights. The policy *is* the network.
- An **action** is sampling one token from that distribution.
- A **trajectory** is a full rollout: start from the question, sample token after token until the model emits `<|assistant_end|>`. That sequence of tokens is one complete attempt at the problem.
- The **reward** is a single number we hand back *after* the trajectory finishes, scoring how good the attempt was.

For GSM8K the reward is dead simple. From `tasks/gsm8k.py`:

```python
def reward(self, conversation, assistant_response):
    is_correct = self.evaluate(conversation, assistant_response)
    return float(is_correct)   # 1.0 if final number matches, else 0.0
```

That's it. We parse the number after the `####` marker out of the model's completion, compare it to ground truth, and return `1.0` or `0.0`. Notice what we are **not** doing: we are not checking whether the reasoning matches, not scoring partial credit, not looking at style. Pure outcome. The model is free to reason however it likes; we only grade the final answer. This is the thing SFT structurally could not do.

> **A note on what makes a good reward.** GSM8K is a gift: the answer is a number, so it's *automatically verifiable*. We can check correctness with a regex, no human and no second model in the loop. This is why math and code are the poster children of LLM RL — the reward is cheap, objective, and un-gameable. The moment your reward needs a human or a learned reward model to judge it, everything gets harder and more expensive. nanochat sticks to the easy, verifiable case on purpose.

---

## The core question: which tokens get credit?

Say we let the model attempt a problem and it produces a 200-token solution that arrives at the right answer. Reward = 1.0. 

Now what? We want to "make this work more likely." But the model didn't make one decision — it made 200 decisions, one per token. Which of those 200 tokens deserve credit for the success? The early ones that set up the approach? The arithmetic in the middle? The final number?

This is the **credit assignment problem**, and it's the central difficulty of RL. The reward arrives once, at the very end, but it has to be distributed back over every decision that led to it.

The beautiful, almost-too-simple answer that **policy gradients** give: *don't try to be clever about it. Give every token in the trajectory the same credit — the trajectory's total reward.* If the rollout was good, nudge **all** of its tokens to be more likely. If it was bad, nudge them all down. Average over enough rollouts and the tokens that systematically lead to good outcomes float up, while the ones that don't wash out as noise.

Let's make that precise.

---

## Deriving the policy gradient (the one piece of math)

### Step 0 — what a "gradient" means here

If you've done SFT you already know the training loop: compute a loss, call `.backward()`, the optimizer takes a step. The gradient ($\nabla_\theta$) is just a long vector — one number per weight in the model — that says "increase this weight to decrease the loss." Gradient *descent* follows that direction to reduce loss.

Here we want to *maximize* reward instead of minimize loss, so we follow the gradient in the *opposite* sign convention — **gradient ascent**. Mechanically it's identical: compute a gradient, step in it. The math below is about deriving *what that gradient is*.

---

### Step 1 — the objective: maximize expected reward

When we say "maximize expected reward," concretely it means: over many different problems and rollouts, we want the average reward to be as high as possible. In math, if $\tau$ is one rollout (trajectory) and $R(\tau)$ is its reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \big]$$

$\mathbb{E}[\cdot]$ just means "average over many samples." The subscript $\tau \sim \pi_\theta$ says those samples are drawn from our current policy (the model with weights $\theta$).

We want $\nabla_\theta J(\theta)$ — the gradient of this average reward with respect to every weight.

---

### Step 2 — a simpler idea first, and why we want more

You might think: out of 4 rollouts, if 2 got the answer right (reward=1) and 2 didn't (reward=0), why not just do SFT on the 2 correct ones? Ignore the failures, treat the successes as gold demonstrations, backpropagate normally.

This is a real algorithm — called **rejection sampling fine-tuning (RFT)** — and it works. In fact, with a pure 0/1 reward, REINFORCE (no baseline) is mathematically identical to it: a $R=0$ trajectory multiplies the gradient by zero, so it contributes nothing, and you're left reinforcing only the correct ones.

**But there's a better version.** The failures aren't just noise to be ignored — they're information. A rollout that got the wrong answer is actively telling you "don't do this." If you could also *push down* the probability of incorrect responses, you'd use the full signal from every rollout, not just half of it.

That requires expressing the objective in a way that handles both directions — reinforcing successes and suppressing failures — cleanly and in a single framework. A mathematical trick called the **log-derivative trick** gives us exactly this, and it's what connects to the advantage-weighted gradient that powers GRPO in the next section.

---

### Step 3 — the log-derivative trick (one algebraic identity)

Before the algebra, let's make one piece of notation concrete: **what is $p_\theta(\tau)$?**

It's the probability that your model (with weights $\theta$) generates that specific sequence of tokens. The model picks one token at a time, each with some probability. The probability of the whole trajectory is the product of every token's probability given everything before it:

$$p_\theta(\tau) = \pi_\theta(\text{"The"} \mid \text{question}) \times \pi_\theta(\text{"answer"} \mid \text{question, "The"}) \times \cdots \times \pi_\theta(\text{"10"} \mid \text{everything before})$$

Same idea as rolling two dice: probability of rolling 3 then 5 = $\frac{1}{6} \times \frac{1}{6}$. Multiply the individual probabilities.

This gives us **two distinct objects, and it's worth keeping them straight** (they look similar but mean different things):

- **$\pi_\theta(a_t \mid s_t)$ — the policy.** The per-token distribution the network outputs at one step: given the context $s_t$, the probability of each possible next token $a_t$. This is literally the model's softmax. It's *the* fundamental object — when we say "the policy," this is it.
- **$p_\theta(\tau)$ — the trajectory probability.** A single number for the *whole* rollout, obtained by multiplying the policy's per-step probabilities together: $p_\theta(\tau) = \prod_t \pi_\theta(a_t \mid s_t)$.

So $p_\theta$ is *built from* $\pi_\theta$ — they're related by that product, not the same thing. (In general RL the trajectory probability would also include environment-transition terms, but for an LLM the "environment" is deterministic — appending the chosen token simply *is* the next state — so the product is purely policy terms.)

The $\theta$ subscript on both is a reminder that these probabilities change when the weights change — that's what training does. A gradient step adjusts $\theta$ to make $p_\theta(\tau)$ higher for good trajectories and lower for bad ones.

---

There's a clever identity from calculus. For any function $p(\tau)$:

$$\nabla_\theta \, p(\tau) = p(\tau) \cdot \nabla_\theta \log p(\tau)$$

*Why is this true?* Chain rule. $\log p$ is a function of $p$, so $\frac{d}{d\theta} \log p = \frac{1}{p} \frac{dp}{d\theta}$, rearranged: $\frac{dp}{d\theta} = p \cdot \frac{d \log p}{d\theta}$. That's it. High school chain rule.

*Why is this useful?* First, a quick note on notation: $\mathbb{E}[f(\tau)]$ (expectation) and $\int p(\tau) f(\tau) \, d\tau$ (integral) are the same thing written two ways. Expectation *is* a weighted average, and in math a weighted average over a continuous space is written as an integral. So:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau) \, R(\tau) \, d\tau$$

Same object, two notations. The integral form is useful here because it lets us write out the gradient explicitly. Now watch what happens:

$$\nabla_\theta J(\theta)
= \nabla_\theta \int p_\theta(\tau) \, R(\tau) \, d\tau
= \int \nabla_\theta p_\theta(\tau) \, R(\tau) \, d\tau$$

The second step moves the $\nabla_\theta$ inside the integral — valid because $R(\tau)$ doesn't depend on $\theta$ (the reward is fixed once the rollout is done), so it's just a constant being pulled outside.

Now apply the identity to swap $\nabla p$ for $p \cdot \nabla \log p$:

$$= \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \, d\tau$$

And now flip back to expectation notation — an integral weighted by $p_\theta(\tau)$ is exactly the definition of $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$. (The subscript $\tau \sim \pi_\theta$ is the conventional shorthand for "trajectories produced by rolling out the policy $\pi_\theta$" — which is sampling from $p_\theta$. People name it by the policy because the policy is the thing you actually control.)

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \big]}$$

The sampling is now *inside* the expectation and we only need to differentiate the $\log p_\theta(\tau)$ term — which is differentiable. The trick moved the differentiation off the sampling operation and onto the log-probability, which the model computes directly. (And in the next step we'll crack $\log p_\theta(\tau)$ open into the per-token policy terms.)

---

### Step 4 — breaking the trajectory into tokens

The boxed result has a $\log p_\theta(\tau)$ — the log-probability of the *whole trajectory*. But the network doesn't output trajectory probabilities; it outputs per-token policy probabilities $\pi_\theta(a_t \mid s_t)$. So we crack the trajectory term open using the relationship from Step 3:

$$p_\theta(\tau) = \prod_{t} \pi_\theta(a_t \mid s_t)$$

where $a_t$ is the token sampled at step $t$ and $s_t$ is everything before it (the context). Taking the log of a product converts it to a sum (the standard $\log(ab) = \log a + \log b$):

$$\log p_\theta(\tau) = \sum_{t} \log \pi_\theta(a_t \mid s_t)$$

This is the key move: the trajectory log-prob is just the **sum of the per-token policy log-probs** — and those are exactly what the network produces. The gradient of a sum is the sum of the gradients:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

---

### Putting it together: REINFORCE

Substituting back in:

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[ R(\tau) \cdot \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

Read this one piece at a time:

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — the gradient that would make token $a_t$ *more likely* at step $t$. This is the direction to nudge the weights to assign higher probability to the token that was actually sampled.
- Multiply that direction by $R(\tau)$ — the reward for the whole trajectory.
- Sum over every token $t$ in the rollout.
- Average over many rollouts.

In plain English:

> **For every token in the rollout: compute the direction that makes that token more likely, then scale it by the trajectory's reward. If the rollout was good ($R=1$), push all its tokens up. If the reward was zero, do nothing. Repeat over many rollouts.**

That's it. That's REINFORCE. The gradient is computable because $\log \pi_\theta(a_t \mid s_t)$ is just the log-probability the model assigns to the token it sampled — something the model computes directly and can backpropagate through.

---

### The connection to SFT that should click right now
For an autoregressive decoder (like the GPT here), the cross-entropy loss over one sequence is:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(a_t \mid a_{<t})$$

where $a_t$ is the true/target token at position $t$, and $a_{<t} = a_1, \dots, a_{t-1}$ is everything before it (the context). Usually you average instead of sum:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(a_t \mid a_{<t})$$


In SFT the training loss for one token is:

$$\mathcal{L}_\text{SFT} = -\log \pi_\theta(a_t \mid s_t)$$

Minimizing that is identical to maximizing $\log \pi_\theta(a_t \mid s_t)$, which is identical to the gradient direction above. So the per-token gradient in RL is **exactly the same computation as SFT**. The one difference:

```
SFT gradient:    gradient of log π  ×  1           (constant, always reinforce)
RL gradient:     gradient of log π  ×  R(τ)        (scale by reward — varies per rollout)
```

SFT is the special case where every demonstration gets reward = 1 (it's shown as correct by assumption). RL lets the reward vary: good rollouts get reinforced strongly, bad rollouts get suppressed, and the signal is proportional to outcome quality. Same engine, smarter throttle.

---

## The variance problem, and the one trick that fixes it

REINFORCE works. But if you actually run it, it learns *slowly* and the training curve looks like a seismograph. The culprit is **variance**, and there's a one-line fix that is genuinely one of the prettiest tricks in RL. Let's see the problem first, because once you see it the fix is obvious.

Here's the thing that should bother you about the gradient we just derived. On GSM8K the reward is always 0 or 1 — **never negative**. So trace what the gradient actually does to each rollout:

- Correct rollout ($R=1$): push all its tokens **up**.
- Wrong rollout ($R=0$): multiply by zero → do **nothing**.

Stare at that for a second. **You never push anything down.** Every single step is some flavor of "make these sequences more likely," and the *only* thing separating a brilliant rollout from a mediocre one is how *hard* you shove it up. The model is supposed to deduce which behaviors are good purely from the fact that the good ones get shoved up slightly more often than the bad ones. That does work in the limit — the softmax is normalized, so relentlessly upvoting the good stuff implicitly downvotes everything else — but it's a maddeningly indirect signal. It's like coaching chess by only ever saying "nice!" at varying volumes and never once saying "no, not that."

Now make it concrete. Take a question the model already nails 90% of the time. Sample 16 rollouts → ~14 correct, ~2 wrong. The gradient cheerfully reinforces all 14 correct ones at full strength — even though the model *already knows how to solve this*. We're blowing almost the entire gradient budget applauding behavior that's already locked in, while the 2 genuinely informative failures contribute exactly nothing. Huge update, near-zero learning. **That's the variance problem: the size of the gradient has almost nothing to do with how much there actually is to learn.**

The fix is a **baseline**. Instead of scaling by the raw reward, scale by reward *minus some reference value* $b$:

$$\nabla_\theta J = \mathbb{E}\Big[ (R(\tau) - b) \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]$$

Two things make this beautiful.

**First, it's free — you can subtract any $b$ without biasing the gradient.** The intuition: REINFORCE only ever responds to *differences* between rollouts. Shift every reward down by the same constant and you haven't changed which rollouts are above the pack and which are below — you've only moved the goalposts equally for everyone. So the baseline can recalibrate *what counts as good*, but it cannot tug the gradient in any direction.

The formal reason is a tidy identity — **the expected score is zero**, $\mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau)] = 0$ — which makes the entire baseline term $\mathbb{E}[\,b \cdot \nabla_\theta \log p_\theta(\tau)\,]$ vanish. Here's why it's zero (the punchline is *not* that some expectation equals 1 — it's that a probability distribution always integrates to 1 *regardless of $\theta$*, so its gradient is 0):

$$
\mathbb{E}_{\tau}\big[\nabla_\theta \log p_\theta(\tau)\big]
= \int p_\theta \,\nabla_\theta \log p_\theta \, d\tau
= \int \nabla_\theta\, p_\theta \, d\tau
= \nabla_\theta \underbrace{\int p_\theta \, d\tau}_{=\,1\ \text{always}}
= \nabla_\theta\, 1 = 0
$$

The middle step runs the log-derivative trick backwards ($p\,\nabla\log p = \nabla p$); the last step is just "the gradient of a constant is zero."

**Second, it brings back the "down."** Pick $b$ to be roughly the average reward, and suddenly $R(\tau) - b$ is positive for better-than-average rollouts and *negative* for worse-than-average ones. Now the gradient pushes the good ones up **and the bad ones down** — both halves of the signal, finally. The quantity $R(\tau) - b$ gets a name: the **advantage**. It stops asking "was this rollout good?" and starts asking the only question that actually teaches: "was this rollout better or worse than what I'd normally do here?"

Now: what's a good baseline? The cleverest choice — and this is the central idea of **GRPO** (Group Relative Policy Optimization) — is to use *the average reward of other attempts at the same question*.

So the recipe becomes:

1. Take one question.
2. Sample a **group** of $G$ completions for it (nanochat uses `num_samples=16`).
3. Compute each completion's reward.
4. Use the **group's mean reward** as the baseline.
5. Advantage of completion $i$ = $R_i - \text{mean}(R)$.

This is gorgeous because the baseline is *per-question and self-calibrating*. An easy question where 15/16 rollouts succeed has a high baseline (~0.94), so the lone failure gets a large negative advantage ("you blew an easy one — fix that") while the successes get a tiny positive nudge ("nice, but everyone did that"). A hard question where only 1/16 succeeds has a low baseline (~0.06), so that one lucky correct rollout gets a big positive advantage ("you found the needle — do more of that"). The signal automatically focuses on the rollouts that are *surprising relative to their peers*. No value network, no extra model — just sample a bunch and subtract the mean.

In nanochat this is two lines, in `scripts/chat_rl.py`:

```python
mu = rewards.mean()
advantages = rewards - mu
```

That's the entire "GRPO." Let that sink in: the famous algorithm is `rewards - rewards.mean()`.

---

## "GRPO," in quotes

We've actually already built the entire algorithm: REINFORCE + a group-mean baseline. But the code calls it "GRPO" (in scare quotes), and the literature has a zoo of acronyms — PPO, TRPO, GRPO, DAPO — that all sound like prerequisites. They're not. Each one is just **REINFORCE plus a patch for a specific pain**, and nanochat removes most of the patches because its setting never feels those pains. So before we say what got dropped, let's build the family tree from scratch. Then "GRPO in quotes" will read as "REINFORCE with the patches peeled back," which is exactly what it is.

### The family tree: each algorithm just patches the last one's headache

The most useful thing to realize about the acronym soup — REINFORCE, TRPO, PPO, GRPO, DAPO — is that it isn't five rival algorithms you have to learn separately. It's **one** algorithm, REINFORCE, with a chain of patches bolted on, and every patch exists to fix one specific, concrete headache the previous version handed you. Follow the headaches and the whole lineage falls out on its own. So let's walk it forward, and watch for where nanochat decides to hop off the train.

**Where we start: REINFORCE** — the thing we already built. Sample some rollouts, weight each token's log-prob by its advantage, take one gradient step. Done. But it has one property that ends up driving everything downstream: it's strictly **on-policy**. The advantage you computed and the gradient you took are valid *only* for the exact model that generated those rollouts. The instant you take a step, the model is a slightly different model — and the rollouts in your hand are now stale, describing a network that no longer exists. So one batch of rollouts buys you exactly **one** gradient step, and then it's trash. And generating rollouts is the expensive part — you're decoding tokens one at a time, across many samples. Burning all that compute for a single step *stings*. That's the first headache, and it's the one that kicks off the whole chain.

**Pain #1: "rollouts are expensive — can't I get more than one step out of a batch?"**
The naive move is to just keep stepping on the same data. But that's quietly broken: after the first step the model has moved, so steps two, three, four are training on samples drawn by an *older* model. The gradient is now biased — those samples don't reflect what the current model would do. The fix is **importance sampling**, which is a fancy name for an obvious correction: reweight each sample by how much more (or less) likely it is *now* than when you drew it,

$$\text{ratio} = \frac{\pi_\text{new}(a)}{\pi_\text{old}(a)} \quad\text{— "how much more likely is this exact token under the current model than under the one that sampled it?"}$$

Scale the gradient by that ratio and the bias cancels out — now one batch of rollouts is good for several steps, not one. Headache #1, patched.

It's worth being precise about how this relates to the REINFORCE objective we derived, because the later acronyms all operate on this form. Our objective was $\log \pi_\text{new}(a) \cdot A$ — a **log-prob** times advantage, valid only when $a$ was sampled from the *current* policy. Importance sampling turns it into a **ratio** times advantage:

$$\underbrace{\log \pi_\text{new}(a)\cdot A}_{\text{REINFORCE (on-policy)}} \quad\longrightarrow\quad \underbrace{\frac{\pi_\text{new}(a)}{\pi_\text{old}(a)}\cdot A}_{\text{surrogate (off-policy, reusable)}}$$

So `ratio · A` is *not a new objective* — it's the off-policy generalization of REINFORCE, built so that its gradient at the moment $\pi_\text{new}=\pi_\text{old}$ (ratio $=1$) is **exactly** the REINFORCE gradient. (Quick check: $\nabla\frac{\pi_\text{new}}{\pi_\text{old}}A = \frac{\pi_\text{new}}{\pi_\text{old}}\nabla\log\pi_\text{new}\,A$, which at ratio $=1$ is just $\nabla\log\pi_\text{new}\,A$.) The only reason to carry the ratio rather than the log is reuse: the log form silently breaks the moment your data comes from an older model, while the ratio form stays unbiased. And note $A$ is a **detached constant** here — the gradient flows only through $\pi_\text{new}$, never through $\pi_\text{old}$ or the advantage. This `ratio · A` is exactly what PPO will clip in a moment.

Where do these two numbers come from? Both are the probability of the **same already-sampled token $a_t$**, evaluated under two versions of the weights. $\pi_\text{old}(a_t)$ is *logged at sampling time* — when Phase 1 draws the token, you record the probability the sampling model gave it. $\pi_\text{new}(a_t)$ is *recomputed at update time* — after a gradient step has moved the weights, you re-run the same token through the current model (the Phase 2 teacher-forced pass) and read off its new probability. The ratio asks "has this exact choice gotten more or less likely since I sampled it?" — and note that on the *first* step, before any update, $\pi_\text{new} = \pi_\text{old}$ so the ratio is exactly 1. It only departs from 1 on the second-and-later reuse steps, which is precisely when it can blow up.

**Pain #2: "okay, but now that ratio can explode."**
Notice the patch created its own problem. As soon as the current model drifts away from the sampling model, the ratio stops being near 1. A token the old model thought unlikely but the new one suddenly loves gives you a *huge* ratio, the gradient's variance detonates, and a single bad batch can wreck the model. So we need to stop the policy from wandering *too far* from where it started in one round of reuse. But to even say "too far," we first need a ruler for "how different are two policies" — and that ruler is the **KL divergence**.

> **Quick aside — what is "KL"?** Everyone uses it to mean "how different are two probability distributions," and that's all it is. Say at one position the old policy thinks the next token is `{cat: 0.6, dog: 0.4}` and the new one thinks `{cat: 0.5, dog: 0.5}`. KL gives you a single number for how far apart those two opinions are — `0` if identical, growing as they diverge. The formula is just a weighted average of the log-ratio of the two probabilities:
>
> $$\text{KL}(p \,\|\, q) = \sum_{v \in V} p(v)\,\log\frac{p(v)}{q(v)}$$
>
> Read it as: for each possible token $v$, look at how many times more (or less) likely $p$ thinks it is versus $q$ — that's $\log\frac{p(v)}{q(v)}$ — and average those log-ratios, weighting by how often $p$ actually produces that token. If $p$ and $q$ agree everywhere, every ratio is 1, every log is 0, and the sum is 0.
>
> For an LLM the "distribution" is the next-token softmax, so you compute this sum **over the whole vocabulary at one position**, then **add it up across all positions** in the sequence to get the KL for the whole response. (Computing the full vocab sum for every token, for two models, is pricey — so in practice people estimate it from just the *sampled* token's probabilities, but the exact definition is the sum above.)

So now we've got a ruler for policy drift. How do you actually keep that drift small? Two historical answers:
- **TRPO** (Trust Region Policy Optimization): here's the concrete problem it solves. You've got a gradient; now you have to decide how far to step along it — that's your learning rate. But a fixed learning rate is treacherous. The same step that's harmless early in training can, a few updates later, shove the next-token distribution off a cliff — the model suddenly starts repeating one token or spewing garbage — because near certain weight settings a small nudge flips the probabilities hard. There's no single step size that's always safe.

  TRPO's move: stop budgeting your step in *weights* and start budgeting it in *behavior*. Pick a KL ceiling — say "the new policy's token distributions may differ from the old by at most $\text{KL} = 0.01$." Each update, walk along the gradient direction but only as far as that 0.01 of KL allows. The annoying part is that "how many weight-units equals 0.01 of KL" isn't constant — some directions barely move behavior, others wreck it — so TRPO has to estimate, right at the current point, how fast KL grows as you move (its local curvature) to find where the 0.01 line sits. It steps to that line, then **actually re-runs the new policy on the batch and measures the real KL**; if it overshot 0.01, or the objective didn't actually improve, it **halves the step and checks again**. That step → measure → halve loop is literally what runs.

  Estimating that curvature is the expensive, second-order bit, and it's exactly why people reached for PPO's cheaper clip instead. (The cheaper *soft* version of TRPO: don't enforce a hard ceiling at all — add a penalty term that grows the further the policy drifts, a spring pulling it back, and dial the spring's stiffness to keep KL near a target. That spring is the KL-to-reference leash in Pain #3.)
- **PPO** (Proximal Policy Optimization): the cheap, wildly popular approximation, and the reason TRPO is a museum piece. PPO throws out all the curvature math and asks a blunt question: instead of carefully measuring drift and sizing the step, what if we just **refuse to let any single token's probability move more than ~20%** per batch? No second-order anything — just clamp the ratio. Recall the plain importance-sampling objective is `ratio · A` (advantage times "how much more likely is this token now"). PPO replaces it with a clipped version:

  $$\mathcal{L}^{\text{PPO}} = \min\Big(\underbrace{\text{ratio} \cdot A}_{\text{normal}},\ \underbrace{\text{clip}(\text{ratio},\,1-\epsilon,\,1+\epsilon)\cdot A}_{\text{ratio frozen to }[0.8,\,1.2]}\Big),\qquad \epsilon \approx 0.2$$

  The two pieces and the `min` look fiddly, so here's the whole thing in concrete terms — it's just "stop rewarding the model once it has already moved a token far enough":
  - **Good token ($A>0$, we want its prob up):** as you raise the prob, `ratio` climbs past $1.2$. The clipped term freezes at $1.2 \cdot A$ — flat, zero gradient — so past +20% there's no more incentive to keep pushing. Below +20%, the clip isn't active and you train normally.
  - **Bad token ($A<0$, we want its prob down):** as you lower the prob, `ratio` falls toward $0.8$, where the clipped term freezes at $0.8 \cdot A$ — again flat, zero gradient — so you can't crush a token's probability to zero in one batch.

  The `min` is the sneaky-important part: it makes PPO always take the *more pessimistic* of the clipped and unclipped values, so clipping can only ever **remove** incentive to move, never add it. That one-sidedness is what stops the model from exploiting the clip to sneak in an oversized beneficial step. The net effect: each token's probability can drift at most ~20% before its gradient is switched off, which loosely bounds how far the whole policy can move — a dirt-cheap, first-order stand-in for TRPO's KL trust region, with no KL computed and no line search.

  And the payoff is the data reuse we wanted back in Pain #1: with the clip as a guardrail, you can safely run **several gradient epochs over the same batch of rollouts** (typically a handful of minibatch passes), then throw the batch out and resample. That's PPO's whole value proposition — squeeze many safe steps out of each expensive batch of rollouts. It's the workhorse of RLHF for exactly this reason.

**Pain #3: "even with safe-sized steps, the model slowly learns to cheat."**
Bounding the *step size* stops blow-ups, but it doesn't stop a slower failure: over many steps, a policy chasing a loose or learned reward will happily creep toward degenerate text that scores high and reads like nonsense (the classic "reward hacking"). The standard leash is a **KL-to-reference penalty**: keep a *frozen copy* of the original SFT model and add a loss term that punishes the policy for drifting too far from *it*. Heads up — this is a different KL from TRPO's, and people mix them up constantly: TRPO bounds the gap between *consecutive* policies (step to step), while this one anchors to a *fixed* reference for the entire run. One limits how fast you move; the other ties you to base camp.

**Pain #4 — the one GRPO actually cares about: "PPO drags around a whole second network."**
Remember the baseline we subtract to get the advantage? PPO doesn't get it for free — it trains a separate **value network** (a "critic") whose job is to predict the expected reward, and that critic is extra parameters, extra forward passes, extra memory, extra things to tune. **GRPO's** one real contribution is to *delete the critic entirely*: instead of learning a value function, just sample a **group** of completions for each question and use the group's mean reward as the baseline. That's the `rewards - rewards.mean()` we already built — the baseline falls out of the samples you were taking anyway, no second network required. GRPO otherwise keeps PPO's clip and usually the KL-to-reference leash.

**And finally DAPO** is a grab-bag of further GRPO tweaks; the only one nanochat borrows is **token-level loss normalization** (explained below).

So the lineage, in one line:

```
REINFORCE  →  +importance ratio (reuse data)  →  +clip = PPO (safe reuse)
           →  +group baseline = GRPO (drop the critic)  →  DAPO (more tweaks)
```

### So what does nanochat actually keep?

Here's the punchline: nanochat is strictly on-policy with **one gradient step per batch** — it never reuses rollouts. And once you never reuse data, *Pains #1–#3 simply don't arise*: there's no old policy to correct for, no ratio to explode, and (for short runs on a clean verifiable reward) no time to game anything. So nanochat keeps only GRPO's actual idea — the group-mean baseline — and throws the rest back. Concretely, here's what got dropped relative to textbook GRPO/PPO, and *why each is safe to drop here*:

**1. No KL-to-reference penalty / no reference model (Pain #3).**
No frozen copy of the SFT model, no KL leash. For short GSM8K runs with a clean, verifiable 0/1 reward there's nothing to game and no time to drift into gibberish, so the regularizer isn't worth its memory and compute cost.

**2. No PPO ratio + clip (Pains #1 and #2).**
The clip only earns its keep when you *reuse* rollouts across multiple gradient steps. nanochat is strictly on-policy — one step per batch, then resample — so the new policy and the sampling policy are identical, the ratio is exactly 1, and the clip would never trigger. Carrying it would be dead weight. The cost of this choice is sample efficiency (every step needs fresh, expensive rollouts); the payoff is that a whole layer of machinery evaporates.

**3. Token-level normalization (DAPO style), not sequence-level.**
When we average the loss, we divide by the **total number of valid tokens across the batch**, not by sequence and then across sequences. The practical effect: a single token in a long rollout carries the same weight as a single token in a short one. Sequence-level averaging would secretly upweight tokens in short responses, biasing the model toward terseness. Token-level keeps it honest.

**4. Plain mean-subtraction, not z-score.**
Classic GRPO standardizes the advantage: $(R - \mu)/\sigma$. nanochat uses just $(R - \mu)$. Dividing by the group's standard deviation turns out to inject its own subtle bias (it inflates the gradient on questions where rollouts happen to agree with each other), so dropping it is arguably *more* correct, not just simpler.

What's left after all that subtraction is essentially **REINFORCE with a group-mean baseline and token-level loss normalization**. Which is the point: nanochat is teaching you that the fancy acronym is a thin wrapper around a very old, very simple idea.

---

## The full loop, end to end

Before the code, here's the shape of the whole thing. RL is an **outer loop** that alternates two phases — **rollout** (generate experience) and **learn** (one gradient step on it) — and then immediately goes back for fresh experience:

```
start from the SFT model
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1 — rollout (no gradients):                       │
│     pick a batch of questions                            │ ◄──┐
│     for each question, sample a GROUP of completions     │    │
│     reward each completion                               │    │
│     advantage = reward − (mean reward of ITS OWN group)  │    │
│                                                          │    │  on-policy:
│  PHASE 2 — learn (gradients on):                         │    │  data is
│     teacher-force the sampled tokens back through         │    │  re-sampled
│     the model, scale each token's cross-entropy by        │    │  every step
│     its advantage, accumulate, and take ONE optimizer step│    │
└──────────────────────────────┬───────────────────────────┘    │
                               └──────── updated model ──────────┘
```

The single most important thing to internalize before reading the code: **once Phase 1 finishes sampling, a rollout is just a frozen list of token IDs — and Phase 2 treats it exactly like an SFT example.** Same teacher forcing, same shift-by-1, same cross-entropy. The model is trained to predict *its own samples*; the only twist is that each token's loss is scaled by the rollout's advantage. There is no separate ground-truth target in RL — the model's own generation *is* the target. (More on this in Phase 2.)

Now let's walk the actual code path in `scripts/chat_rl.py`.

### Phase 1 — Rollout: generate experience (`get_batch`, under `@torch.no_grad()`)

```
for each training question:
    1. render the prompt, but DROP the reference answer
    2. sample 16 completions at temperature 1.0
    3. score each completion -> reward in {0, 1}
    4. advantage = reward - mean(reward)
    5. yield (tokens, inputs, targets, rewards, advantages)
```

**Step 1 — prime the model for a completion.** We take the conversation and strip the assistant's gold answer off the end, keeping only the `<|assistant_start|>` token so the model is poised to write its own attempt. From `nanochat/tokenizer.py`:

```python
def render_for_completion(self, conversation):
    ...
    messages.pop()                       # remove the gold assistant message
    ids, mask = self.render_conversation(conversation)
    assistant_start = self.encode_special("<|assistant_start|>")
    ids.append(assistant_start)          # prime for completion
    return ids
```

This is the crucial difference from SFT. In SFT the gold answer is present and we compute loss against it. In RL **the gold answer is deleted** — the model has to produce its own, and we'll grade it. The training data is only ever used as a question bank and an answer key for the reward function; the model never imitates the reference solution.

**Step 2 — sample a group of rollouts.** We ask the inference `Engine` for 16 completions at `temperature=1.0`. Temperature matters: at temp 0 the model is greedy and all 16 rollouts would be near-identical — no diversity, no group to compare against, no learning signal. We *want* exploration here, so we sample hot. The engine (`nanochat/engine.py`) does a single prefill of the shared prompt, clones the KV cache 16 ways, and decodes all rollouts in parallel. It also runs the **tool-use state machine** mid-rollout: when the model emits `<|python_start|>...<|python_end|>` the engine actually executes the calculator and *forces* the result tokens back into the stream.

**Step 3 — reward.** Decode each completion, run `task.reward(...)`, collect a `(16,)` vector of 0s and 1s.

**Step 4 — advantage.** `advantages = rewards - rewards.mean()`. The crucial detail: `get_batch` computes this **one question at a time**, so `rewards.mean()` is the mean over *that question's own group of completions* — never pooled across different questions. Each group is normalized against its own baseline. A consequence worth knowing: if every completion in a group gets the *same* reward (all correct, or all wrong), the mean equals every reward, so **all advantages are zero** and that question contributes *no gradient at all* this step. Small groups go unanimous more often, which is one reason the group size (`num_samples`) is a real knob — too small and a chunk of your questions silently go dead each step.

### Phase 2 — Learn: the policy-gradient step

First, where do the `inputs` and `targets` come from? **Both are carved out of the sampled rollout itself** — back in `get_batch`:

```python
ids     = torch.tensor(padded_generated_token_sequences, ...)  # the tokens the model SAMPLED
inputs  = ids[:, :-1]          # shift
targets = ids[:, 1:].clone()   # shift by 1 — teacher-force against the model's OWN output
```

This is the SFT teacher-forcing setup *exactly*: shift the sequence by one, predict each token from everything before it. The only difference from SFT is that the target tokens aren't a human-written gold answer — they're the tokens the model sampled for itself in Phase 1. We re-run them through the model (this time **with gradients on**, unlike the `no_grad` sampling pass) so we can get $\nabla_\theta \log \pi_\theta$ for each sampled token and actually backprop. With that in hand, the policy-gradient step is startlingly short:

```python
# logp of every token. The model's loss is NLL = -logp, so we negate to get logp.
logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)   # (B, T)

# policy-gradient objective: scale each token's logp by its rollout's advantage
pg_obj = (logp * advantages.unsqueeze(-1)).sum()

# token-level normalization (DAPO): divide by number of valid tokens
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

# we maximize the objective -> minimize its negation
loss = -pg_obj
loss.backward()
```

Map this line by line back to the math:

- `model(inputs, targets, loss_reduction='none')` returns the per-token cross-entropy = per-token NLL = $-\log \pi_\theta(a_t \mid s_t)$. We negate to recover $\log \pi_\theta$. Notice the `loss_reduction='none'` — we need the loss *per token*, not averaged, because each token will be scaled by an advantage. (See `nanochat/gpt.py`, where `F.cross_entropy(..., reduction=loss_reduction)` makes this switch possible.)
- `logp * advantages.unsqueeze(-1)` is exactly $A_i \cdot \log \pi_\theta(a_t \mid s_t)$ — broadcast the per-rollout advantage across all that rollout's tokens. This is the REINFORCE term with the group baseline baked into $A$.
- `.sum()` over all tokens and rollouts.
- Divide by the valid-token count for token-level normalization.
- Negate, because PyTorch optimizers *minimize* and we want to *maximize* expected reward.

That's the whole gradient. No value head, no GAE, no clipping. Just "scale next-token cross-entropy by advantage."

### The masking detail that's easy to miss

Look again at how targets are built in `get_batch`:

```python
targets = ids[:, 1:].clone()
targets[mask_ids[:, 1:] == 0] = -1   # -1 is the ignore_index
```

The engine returns a **mask** that is `0` for two kinds of tokens: (a) the prompt tokens, and (b) tool-use tokens that the engine *forced* into the stream (the calculator results). Setting those targets to `-1` makes `cross_entropy(ignore_index=-1)` skip them entirely.

Why this matters, and why it's correct:

- We must not train on the **prompt** — the model didn't choose those tokens, so they carry no policy-gradient signal.
- We must not train on the **forced tool-output** tokens — *the model didn't sample those either*; the engine injected them. Reinforcing tokens the model never chose would be teaching it to take credit for the calculator's work. We only apply the policy gradient to tokens the model actually **sampled itself**. That's the only place the $\log \pi_\theta$ term is meaningful.

Get this wrong and you'd be quietly corrupting the gradient with tokens that have no business being in the objective. It's a one-liner, but it's load-bearing.

### Accumulate, step, repeat — and what "one step" actually means

A **step** here means exactly one thing: **one call to `optimizer.step()` — one weight update.** All the rollouts gathered this step pour their gradients into that single update, the weights move once, and then we throw the rollouts away and sample fresh ones from the now-updated model. That last part is what makes it *on-policy* (and why there's no PPO clip): every step trains on data the current model just produced.

There are two separate "batch size" knobs, and conflating them is the usual source of confusion:

- **`examples_per_step`** (default 16) — how many *questions* feed into one weight update. This sets how broad a spread of problems informs each step.
- **`device_batch_size`** (default 8) — purely a *memory* limit: how many sequences fit through a single forward pass. It has nothing to do with the math.

If the total sequences in a step (`examples_per_step × num_samples`) exceed `device_batch_size`, the loop just splits them into chunks and **accumulates gradients across chunks before the one `optimizer.step()`**. So "one step" can be many forward/backward passes under the hood — but still exactly one weight update. The chunking is bookkeeping; it never changes the result.

> **Worked example.** Say you wanted 2 questions per step with a group of 4 (so 8 sequences total) and a `device_batch_size` of 8: sample 2 questions × 4 completions, reward all 8, compute each *group's* advantage, run forward/backward on all 8, call `optimizer.step()` **once**, zero the grads, then pick 2 *brand-new* questions and repeat. That whole cycle is one step.

A few more practical touches worth noting:

- **Tiny learning rate.** `init_lr_frac=0.05` starts RL at 5% of the SFT learning rate, then ramps linearly to zero. RL gradients are noisy and the SFT model is already good — you want gentle nudges, not bulldozing.
- **No optimizer state saved.** The checkpoint deliberately skips Adam/Muon moments (`save_checkpoint(..., None, ...)`). RL runs are short and on-policy; it's not worth the disk.
- **`fp16` not supported for RL** (per the repo README) — the gradient-scaler dance used in base training isn't wired up here. Use `bf16`/`fp32`.

---

## How do we know it's working? `pass@k`

Every `eval_every` steps we run the held-out test set and compute **pass@k**: out of $k$ sampled attempts at a problem, did *at least one* get it right? (See `run_gsm8k_eval`.)

```
pass@1  -> model is right on a single try
pass@8  -> at least one of 8 tries is right
```

This is the right lens for RL, and the relationship between the two numbers tells a story:

- **`pass@k` (large k) is your ceiling** — it measures whether the *capability* to solve the problem exists anywhere in the policy's distribution. Pretraining + SFT mostly set this.
- **`pass@1` is your floor** — it measures whether the model reliably *commits* to the right path on the first try.

The whole job of this flavor of RL is to **drag `pass@1` up toward `pass@k`**. The model already *can* solve these problems sometimes (that's why some rollouts get reward 1). RL doesn't teach a brand-new skill so much as it **sharpens the distribution onto the reasoning paths that already work**, making the good outcome the *likely* outcome instead of the occasional lucky one. You're not expanding what's possible; you're concentrating probability mass on the parts of the policy that succeed.

---

## The one-paragraph version

You did SFT, which is imitation and therefore capped by your demonstrations. RL lets you optimize the *outcome* directly: let the model attempt a verifiable problem, score the attempt with a reward, and reuse your SFT cross-entropy gradient — but scaled by that reward — so successful rollouts get reinforced and failures suppressed. Sample a group of attempts per question and subtract the group's mean reward to get a low-variance **advantage** (that's the whole of "GRPO"). Strip away the trust region, the PPO clip, the z-score, and the value network — none are needed when you're on-policy with a clean reward — and you're left with REINFORCE plus a group baseline and careful token masking. Watch `pass@1` climb toward `pass@k` and you'll know it's concentrating probability onto the paths that already work.

```
pretraining   ->  learn language          (imitate the internet)
SFT           ->  learn to be an assistant (imitate good answers)
RL            ->  learn to be RIGHT        (try, get scored, reinforce what worked)
```

---

## Where to look in the code

| Concern | File |
|---|---|
| The whole RL training loop | `scripts/chat_rl.py` |
| Reward function (verifiable 0/1) | `tasks/gsm8k.py` |
| Prompt priming (drop the gold answer) | `nanochat/tokenizer.py` → `render_for_completion` |
| Batched rollout sampling + tool-use machine | `nanochat/engine.py` → `generate` / `generate_batch` |
| Per-token loss (`loss_reduction='none'`) | `nanochat/gpt.py` → `forward` |

Run it:

```bash
# 1 GPU
python -m scripts.chat_rl

# 8 GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
```
