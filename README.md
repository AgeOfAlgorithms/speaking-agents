# Speaking Agents

Can small text-decision models learn to **talk to each other** in order to survive together?

This project turns [Crafter](https://github.com/danijar/crafter) (a 2D Minecraft-like survival game) into a
cooperative multiplayer world where players can speak, hear each other within earshot, and share
what they carry — and then puts small language-encoder models in it as players. A human can join
the same game from the browser and talk to them.

> **Status: work in progress.** The world, the playable GUI, the scripted baseline team, zero-shot
> evaluation and the imitation warm-start pipeline exist. Reinforcement learning on top of the
> warm start — the part that tests whether talking actually *emerges as useful* — is not built yet.

## The question

[Laya](https://github.com/NandhaKishorM/laya) and [von](https://github.com/wfzyx/von) are ~400M-parameter
*non-autoregressive decision models*: text goes in, and one forward pass scores a list of options
that are themselves written as text. For a single game with a fixed set of actions that buys
nothing — in an earlier single-player experiment a 1M-parameter MLP matched a behaviour-cloned Laya.
What a text model *can* do that a fixed classifier cannot is take a changing list of options, read
sentences it hears, read teammates' names, and start with meanings for words.

So the experiment is: give agents a small spoken language made of **real English words**, make
speech cost something, and see whether training makes them use it to coordinate — and whether
pretrained word meanings help, compared with the same language with its vocabulary shuffled.

## The world

Crafter's rules, plus:

| | |
|---|---|
| **Players** | 2–4 per world, random names each game; monsters go for the nearest standing player; more players, more monsters at night |
| **Speech** | Saying something **takes your turn**. Anyone within 16 tiles (a circle, wider than the 9×7 view) hears it next step, with who said it and where from. Heard sentences land in the listener's event log |
| **Sharing** | `drop_<item>` puts one item on the tile you face; `use` picks it up |
| **Water buckets** | wood (5 drinks) / iron (12). `use` on water drinks *and* fills; one drink fully quenches; set a bucket down and anyone can drink from it. A night cannot be survived awake on one drink bar, so a camp needs one |
| **Campfires** | wood (3 wood, burns a third of the night) and stone (3 wood + 3 stone, the whole night); 3 wood refills either; a dying fire looks different. Zombies and shades cannot spawn near or step into firelight. A sleeper cannot feed a fire, so someone keeps watch |
| **Shades** | night-only monsters that spawn on cave floor and sand (where zombies do not), hunt the nearest player, dissolve at dawn, and cannot enter firelight. Skeletons ignore fire |
| **Downed, not dead** | at 0 health you are down for 40 steps and can still call for help; a teammate's `use` revives you |
| **Team reward** | each of Crafter's 22 achievements counts **once for the whole team** |

Agents speak a closed-vocabulary language (about 90 words plus the current players' names):
up to 8 words, free word order. A human player may type anything.

## What's here

```
core.py             what a player perceives: 9x7 view, stats, event log, remembered landmarks -> text state
teacher.py          rule-based single-player Crafter brain used by the scripted teammates
mp/engine.py        the multiplayer world and all the rules above
mp/language.py      the vocabulary; compass directions and distance words
mp/agents.py        scripted co-op players (camp, watch, buckets, ask-and-hand-over, rescue), random, human
mp/prompts.py       how the game is put to a text model: instruction + only the actions possible right now
mp/laya_policy.py   Laya and von as zero-shot players; the trained player
mp/model.py         trainable player: Laya's encoder + option scorer, plus a small speech decoder on the same pass
mp/collect.py       record warm-start data from the scripted team
mp/train_bc.py      imitation warm start (checkpoints every 1000 steps, --resume)
mp/evaluate.py      score a team on held-out worlds
mp/run_pipeline.py  baselines -> data -> training -> evaluation, resumable
gui/                local web app: play with the agents, watch them, see what each model was offered and chose,
                    speech bubbles, conversation logs, live training curves
```

## Running it

Windows / Linux, Python 3.11, an NVIDIA GPU for the models (the game and scripted agents need none).

```sh
conda create -n laya python=3.11 -c conda-forge
conda activate laya
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install crafter "transformers>=4.48" safetensors huggingface_hub numpy pillow imageio einops
pip install laya                      # on Windows set PYTHONUTF8=1 first

python -m mp.test_engine              # rule tests
python gui/server.py                  # http://127.0.0.1:8765  (run_gui.bat on Windows)
python -m mp.evaluate --team scripted --games 50
python -m mp.evaluate --team laya --games 20          # zero-shot Laya; --team von for von
python -m mp.run_pipeline             # baselines -> warm-start data -> train -> evaluate
```

In the GUI: WASD/arrows to move, space to use, Craft / Place / Drop menus grey out whatever is not
possible right now and say why, Enter to chat.

## Results so far

Team score = Crafter's score (geometric mean of achievement success rates) computed on the team,
over held-out worlds. Three players per team.

| team | team score | achievements / game | player lifetime (steps) |
|---|---|---|---|
| random | 1.9 | 2.9 | 178 |
| Laya, zero-shot (sampled / always top choice) | 1.9 / 0.0 | 2.9 / 0.0 | 184 / 212 |
| von, zero-shot (sampled) | 1.7 | 3.9 | 175 |
| **Laya, warm-started by imitation** (full fine-tune, 57k decisions) | **25.9** | 13.0 | 277 |
| **the same, with a speech decoder pretrained to describe what it sees** (below) | **30.5** | 13.1 | 309 |
| Laya, imitation with only the top 8 of 28 encoder layers trainable | 17.0 | 9.9 | 193 |
| Laya, imitation with the encoder frozen | 6.9 | 6.9 | 188 |
| scripted co-op, talking (the team it imitated) | 29.2 | 13.5 | 360 |
| scripted co-op, silent | 28.1 | 13.8 | 341 |
| scripted co-op, speech muted | 28.8 | 13.5 | 297 |

**Zero-shot, neither model can play**: Laya and von are indistinguishable from random, and von costs
one forward pass per option (several times slower) for the same result.

**Imitation works this time.** One pass over 57k recorded decisions takes Laya from random to ~90% of
its teacher's score, agreeing with the scripted team on 80% of held-out decisions. It lights campfires
(2.1 a game), hands items over (7.6 dropped, 6.7 picked up) and fights shades. It does not yet share
water, and its players go down more often than the scripted ones.

**Speech needed its own training phase.** Only 3.6% of recorded decisions are sentences, so after the
main pass the decoder produced fragments ("help" for "help me"; 0% of held-out sentences right). Two
fixes: a decoder-only phase over the spoken samples (`mp/tune_speech.py`; the encoder is frozen, its
outputs cached, 25 epochs take minutes), and a LayerNorm on the encoder states the decoder reads —
raw, they saturated it. Held-out: 86% of words, 67% of whole sentences; the misses are mostly the exact
compass word or distance in "I see coal southwest pretty close". With the same action policy:

| warm-started Laya, speech decoder | team score | revives / game |
|---|---|---|
| fragments | 23.9 | 0.4 |
| first tuning pass (44% of sentences right) | 24.0 | 1.4 |
| tuned + normalised (67%) | 25.9 | 1.5 |

Twenty worlds is a small sample, but the direction is the first sign that the words do something:
a downed player who can say "help me" properly gets picked up.

**Unfreezing the encoder matters.** Held-out agreement with the scripted team / team score: whole
encoder trainable 80% / 25.9, top 8 layers 65% / 17.0, frozen 50% / 6.9.

**Speak first, understand later.** Learning to talk and to understand at the same time is a
chicken-and-egg problem, so the two are separated. `mp/describe.py` computes, for any recorded state,
the sentences in the agents' vocabulary that are TRUE of it ("few zombie south close by", "Pia is east
pretty close", "I have many stone", "night soon", "I can make pickaxe"; about 15 per state), and
`mp/pretrain_speech.py` trains the decoder on 16,000 recorded states to say them, with no game running.
When speaking is worth a turn, and what to do about what one hears, is left to reinforcement learning.

The finding that made it work: the decoder must not read the encoder's final states. After
fine-tuning for action choice those no longer carry details like "3 west 1 north"; a decoder reading
them became fluent but was right about the state only ~40% of the time, however it attended. Reading
encoder layer 16 of 28 instead, the sentence it thinks most likely is true of held-out states 100% of
the time, and sentences sampled from it 68%. (RL trains only the top 8 layers, so what the decoder
reads cannot drift during RL.) A short mixed phase keeps the scripted team's sentences at the scripted
team's moments: 95% of held-out words, 83% of whole sentences (67% before).

With that decoder and the unchanged action policy the team scores 30.5 (25.9 before), with 2.0
revives a game and the first shared drinks from a placed bucket. Caveats: 20 worlds; and two log fixes
went in at the same time (a revive is now logged as one, and where a heard sentence came from is
re-expressed from where the listener stands now), so this is not a clean single-change comparison.

Honest reading of the scripted numbers: with a hand-written protocol, talking makes players live a
little longer and lets them hand each other the makings of a stone campfire, but it does not move the
team score. Whether a *learned* protocol does is the open question; reinforcement learning from the
warm start (speech on vs off) is running: PPO with a KL anchor to the warm start (so speech does not
die out before it pays) and a centralised critic. von's encoder under the same head follows.

## Credits and licences

- The game is built on [Crafter](https://github.com/danijar/crafter) by Danijar Hafner (MIT). The extra
  sprites in `mp/assets/` are generated by `mp/make_assets.py`; the player and dropped-item sprites
  are derived from Crafter's textures.
- [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) and [von](https://github.com/wfzyx/von)
  (Apache-2.0) are downloaded from the Hugging Face Hub at run time; no model weights are in this repo.
- This repository's own code is MIT licensed (see `LICENSE`).
