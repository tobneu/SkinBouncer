# Scope and ethics

SkinBouncer builds classifiers that flag Minecraft player skins for human review. That is
a content-moderation system, and content-moderation systems are wrong about people in ways
that matter. This document is what we think you should know before running one.

It is not a disclaimer. Most of it is design constraints that shaped the code — where they
did, the file is named.

---

## What this is

**A warning system.** A detector produces a score; a threshold turns that score into a
flag; a flag is a prompt for a person to look. That is the whole intended loop.

**Not an auto-banner.** The bundled Paper plugin
([`minecraft_plugin/`](minecraft_plugin/)) deliberately has no kick or ban path. It logs
and it messages holders of `skinbouncer.notify`. The joining player is not told and not
interrupted. You can obviously write a plugin that bans on `risk: true` — the API doesn't
stop you — but nothing about the numbers below supports doing it.

At the operating point this project was tuned for (recall 0.95), the original CNN's
precision was **0.58**. Two out of five flags were wrong. An automatic ban at that
precision is a system that wrongly punishes 42 % of the people it acts on.

---

## Why the demo category is Spider-Man

The categories that actually motivate this — sexual content, hate symbols, harassment
imagery — are ones a university project has no business collecting, storing or
redistributing. Training a detector for them means first assembling a thousand examples of
them, which is the part nobody should be doing for coursework.

So the demo class is **Spider-Man skins**: visually coherent enough to be learnable at
64×64, trivially easy to collect, and harmless to have sitting on a hard drive. That is
the entire justification. It is an arbitrary stand-in — Spider-Man instead of something
nobody wants in a dataset — not a category anyone claims servers are actually policing.

The substitution is honest about what it does and doesn't establish. It shows the
machinery works. It says nothing about how well a model would separate hate symbolry from
superficially similar imagery, which is a far harder problem with far worse consequences
for being wrong.

**If you train on a real prohibited category, the pipeline is unchanged but the stakes are
not.** Everything below applies with more force.

---

## What the model cannot do

It sees a 64×64 RGBA texture. It has no access to anything else, and this is a hard limit,
not a gap to be closed by more training data:

- **No context.** It cannot know whether a symbol is worn in earnest, in parody, in
  historical reference, or in ignorance of what it means.
- **No intent.** Two identical skins worn by two players mean two different things, and
  the pixels are the same.
- **No reappropriation.** Symbols and slurs that a community has reclaimed look exactly
  like the ones it hasn't.
- **No resemblance threshold.** "Looks a bit like the training set" and "is the thing the
  rule prohibits" are different claims, and the score conflates them.

These are precisely the distinctions moderation decisions turn on, and precisely the ones
a texture classifier is structurally unable to make. It is a *prioritizer of human
attention*. Treating its output as a judgment is a category error.

Two concrete failure modes worth expecting:

- **Skin tone and correlated features.** Datasets of player skins are not balanced, and a
  classifier trained on "flagged" examples that happen to correlate with a visual attribute
  will learn that attribute. Nothing in this pipeline detects that for you. Look at your
  false positives, grouped, before you deploy.
- **Whole-texture judgments.** The CNN scores the entire 64×64 image at once, so a large
  benign region can dilute a small prohibited one, and a dominant color scheme can trigger
  on a skin that has nothing else in common with the training set.

---

## The threshold is the policy

`threshold.json` is not a model detail. It is the parameter that decides how much wrongful
suspicion your moderators generate in exchange for how much prohibited content they catch,
and it belongs to whoever is accountable for that trade — not to us, and not to the
training run.

`scripts/train_detector.py` searches it on the **validation** split for a recall target
(default 0.95) and writes it out as a deployment-time value for exactly this reason. Moving
it toward higher recall means more false accusations; moving it toward higher precision
means more prohibited skins pass. There is no setting that avoids the choice, and no
default we could ship that would be right for your server.

Pick it by looking at your own confusion matrix on your own test split
(`scripts/export_detector.py` prints it), not by taking ours.

---

## Data

**Nothing ships with this repo.** No dataset, no trained weights, no scraper. A fresh
clone has code and documentation and nothing else — `.gitignore` covers `data/`,
`sample_data/`, `detector_projects/` and `models/`. This is a deliberate position, not an
oversight:

- Redistributing scraped skin images means redistributing other people's artwork.
- A checkpoint trained on scraped data is a derivative of it.
- A detector should be trained on the policy it will enforce. Ours is not yours.

`scripts/generate_sample_data.py` gets you a runnable dataset without any of that: real
skins from the **official public Mojang API** (the same endpoint the deployment path uses,
no third-party site, no ToS problem), plus a synthetic "flagged" class marked with a
smiley drawn pixel-by-pixel in the script — a self-made marker, so there is no copyright
question and no offensive imagery in the demo path.

The keyword scraper used for the original Spider-Man set was removed from this repo. If
you collect your own data, that site's terms are between you and them.

### Player data at runtime

`POST /check/player/` sends a player name or UUID to Mojang and downloads the current
skin. Two things follow: Mojang learns which players your server is checking, and you are
processing personal data if your jurisdiction counts a player identifier as such. The API
holds each skin in a temporary directory only for the duration of the request and deletes
it afterwards ([`06_Deployment/api/main.py`](06_Deployment/api/main.py)) — it deliberately
keeps no history. If you add logging, you are the one deciding retention.

---

## Before you point this at real players

- **Put authentication or some purposeful protection in front of the API.** It has none. It is marked local-use-only in
  the source and means it — an open endpoint lets anyone score arbitrarFailed to download file.
Name: netty-resolver-4.1.115.Final.jar
URL: https://libraries.minecraft.net/io/netty/netty-resolver/4.1.115.Final/netty-resolver-4.1.115.Final.jar
Error details: Failed to connect to libraries.minecraft.net port 443 after 135753 ms: Could not connect to server
Filename on disk: netty-resolver-4.1.115.Final.jar
Path: /home/tobi/.minecraft/libraries/io/netty/netty-resolver/4.1.115.Final/netty-resolver-4.1.115.Final.jar
Exists: Nonexistenty players against
  your detectors and infer your policy.
- **Curate a real test split**, blind, using `scripts/run_blind_test_review.py`. Metrics
  from a test set you never checked are metrics about your labeling habits.
- **Read your false positives by hand.** Group them. If a pattern appears that isn't the
  concept you meant to detect, you have found the thing your model actually learned.
- **Decide retention before you log.** Scores about identifiable players accumulate into a
  record of suspicion.
- **Give people a way to appeal**, staffed by a human who can see the skin and overrule the
  flag. A flag your moderators cannot overturn is an automated decision no matter what the
  plugin does.
- **Tell your players the system exists.** Covert automated screening is a worse position
  to be in than an announced one, both ethically and in most regulatory regimes.
- **Re-check after you retrain.** The active-learning loop changes labels, and changed
  labels move the threshold's meaning.

---

## Uses we do not support

Fine-tuning this on a category whose purpose is to identify or exclude people by a
protected characteristic — ethnicity, religion, disability, gender identity, sexual
orientation. The pipeline is generic and we cannot stop it; the MIT license grants you
permission and this section does not remove it. But the project is built for enforcing
published server rules about imagery, and building a detector for who someone is rather
than what they display is not a use we'd help with.

---

## Legal

MIT-licensed, provided as-is, with no warranty — see [LICENSE](LICENSE). Nothing here is
legal advice. Whether you may collect skins, how long you may keep scores, and what you
owe players before screening them automatically depends on your jurisdiction and your
platform's terms.
