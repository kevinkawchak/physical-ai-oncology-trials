# Chunk 06 — Related Works, Discussion, Conclusion, and Acknowledgments

---

## 6. Related Works

**Software engineering benchmarks.**
Early evaluations of LMs' coding capabilities typically tasked models with completing the body of a function given its header and a brief description.
As performance on such benchmarks has saturated, the community's attention has shifted towards more complex, repository-level tasks, notably SWE-bench.
Given a GitHub issue, an LM must rewrite the codebase such that the proposed fix passes one or more unit tests.
SWE-bench has since been extended in multiple directions, including evaluation, issue resolution workflows and SWE-agents, and datasets.
Unlike these benchmarks where the objective and often the recommended approach are explicitly specified, CodeClash offers no predetermined notion of what constitutes improved code.
LMs must determine and pursue their own refinement strategies.
This adversarial setting evaluates capabilities beyond codebase manipulation, such as strategic thinking, adaptation to opponents, and long-term planning.

**Performance optimization.**
In lieu of unit tests, several benchmarks instead evaluate LMs on code optimization, such as boosting algorithmic efficiency or reducing runtime.
Like CodeClash, how an LM goes about improving a codebase is entirely self-prescribed; there are no specific instructions or hints about methodology.
Unlike CodeClash, first, LMs carry out optimizations independently; LMs' codebases do not directly compete, nor must LMs anticipate or adapt to opponents' strategies.
Second, the objectives of existing optimization tasks are relatively narrow.
In contrast, CodeClash supports diverse environments with flexible win conditions, enabling LM-based code evolution for goals beyond runtime performance.

**Game playing.**
Video and text games have long been used as testbeds for studying reinforcement learning agents, with a resurgence in use for evaluating LMs.
While past works have an AI system directly play a game, to our knowledge, CodeClash is the first to study the interplay of interactive coding and gaming for evaluating LMs.
Furthermore, CodeClash's task formulation aims to represent not just games, but general real-world, competitive software development, where codebases essentially compete against one another to achieve goals.

**Self improving agents.**
Recent work has explored how LMs can evolve agent scaffolds for better performance on software development tasks, namely SWE-bench.
However, static benchmarks relying on fixed correctness metrics like unit tests are an awkward fit for prototyping self-improvement systems.
Unit tests only provide binary pass/fail feedback, and once passed, they are no longer useful for further refinement.
CodeClash's competitive setting with constantly evolving opponents provides a perpetual learning signal that doesn't saturate.
Performance is graded relatively, a much richer training signal than binary correctness.
We hope future work around self-improving SWE-agents will consider CodeClash as a training ground.

---

## 7. Discussion

**Limitations and future directions.**
CodeClash's code arenas are relatively smaller and more self-contained than most real-world software systems.
We'd be excited to support code arenas encompassing tougher settings, where SWE-agents manage larger codebases attempting to win multiple competitive objectives (e.g., city planning, disaster preparedness, cybersecurity).
Second, CodeClash uses `mini-SWE-agent`, reflecting our intention to focus on the evaluation of LMs by holding the agent scaffold constant.
With that said, a simple next step could be to swap out `mini-SWE-agent` with tool-based frameworks to maximize AI systems' performance.
Third, logs from the competition phase are entirely text-based.
We don't explore Vision Language Models (VLMs) in this work.
Supporting multimodal feedback is on the road-map for future investigations.
Finally, we are curious about the value of CodeClash's artifacts and environments towards improving model capabilities via pre-training on traces of models' edits or post-training with techniques like self-play and reinforcement learning.

**Conclusion.**
By situating LMs in tournaments where their codebases compete directly, CodeClash reveals both the creative potential and fundamental limitations of current models.
Models devise remarkably diverse solutions and demonstrate technical proficiency, but struggle to draw meaningful conclusions from competition logs or maintain well-organized codebases over time.
These findings offer clear avenues for future work.
We hope CodeClash will serve as a reliable, extensible training ground for evaluating and building the next generation of long-running, autonomous software development systems.

---

## Acknowledgments

We thank Laude Institute, Andreessen Horowitz, and Open Philanthropy for providing funding for this work.
We thank Princeton Language & Intelligence (PLI) for providing credits for running closed-source API models.
Thanks to Samuel Ainsworth for his constant support of `bitbop.io` (https://bitbop.io/), the compute service for which this project was carried out with.
We also thank Shiyi Cao, William Held, Abe (Bohan) Hou, Dacheng Li, Jeffrey J. Ma, Karthik R. Narasimhan, Yijia Shao, Chenglei Si, Zora (Zhiruo) Wang, Alexander Wettig, and Yanzhe Zhang for constructive discussions and support throughout this project.
Finally, our greatest thanks to the open source development communities that created and maintain several of the competitive code arenas represented in CodeClash.
