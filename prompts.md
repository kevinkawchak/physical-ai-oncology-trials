# Prompts

Development prompts used to generate content for this repository.

---

## USL Surgical Robot Standard Prompt (v1.5.0)

*Used: February 2026 — Claude Code Opus 4.6*

Your goal is to adapt a comprehensive new "Unification Standard Level" (USL) surgical robot standard (three different manufacturers of the same type) (under main/unification/usl) that both works from and pushes developments to kevinkawchak/physical-ai-oncology-trials (update relevant Readme and other documentation throughout the repo based on your update). New directory: kevinkawchak/physical-ai-oncology-trials/tree/main/unification/usl/surgical. Adapt to the existing a)-d) criteria used for Cobots, the 1-10 scale in 0.1 increments, standards, text diagrams, etc. and surgical robot related code from the existing usl directory.

All new code must be complete and end-to-end with the goal of each of the three surgical robots becoming more unified based on the a)-d) criteria. The kevinkawchak/physical-ai-oncology-trials/tree/main/unification/usl Readme general information must now apply to both surgical robots and the prior cobots information. Move the existing kevinkawchak/physical-ai-oncology-trials/blob/main/unification/usl/usl_scoring_framework.py to under the cobots directory. Create three new text diagrams for the surgical robots in the readme. The readme should contain general, surgical, and cobots details in that order. The surgical robot directory, its corresponding py and its three subdirectories for each of the three surgical robots should adapt to the cobots file structures.

It is important that you clone the most current version of physical-ai-oncology-trials, and that your decisions be based on the current repository and reputable sources of information firstly from other reputable GitHub accounts and secondly from other reputable sources of online information (peer or non-peer reviewed). Cite sources you use properly.

The Readme for main/unification/usl should provide 3 additional different effective text diagrams illustrating general differences between each surgical robot, technical differences between each surgical robot, and scoring differences between them. Each surgical robot should have its own directory with useful and comprehensive code to aid its future unification process with other surgical robot models from both inside their organization and outside their organization. Move and update the prompts.md under main to now include the new full prompt under kevinkawchak/physical-ai-oncology-trials/tree/main/unification/usl.

Be sure to fix and address errors that would cause failed checks for the single pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Provide an updated changelog (v1.5.0). Put this prompt into prompts.md under main. Place the new release notes in a releases.md under main using the format below. Update other relevant documentation such as project structures.

"FORMAT"
Release title
v1.5.0 -

## Summary

## Features

## Contributors
@kevinkawchak
@claude

## Notes

---

## USL Standard Prompt (v1.4.0)

*Used: February 2026 — Claude Code Opus 4.6*

Your goal is to develop a comprehensive new "Unification Standard Level" (USL) standard (under main/unification/usl) that both works from and pushes developments to kevinkawchak/physical-ai-oncology-trials (update relevant Readme and other documentation throughout the repo based on your update). USL in this context is not a phrase widely used, so its context should stay primarily within standardizing and evaluating the unification levels of different types of ai robots (Final scores are 1-10 by 0.1 increments) to be utilized in upcoming physical ai oncology trials based on their its ability to a) switch between simulation frameworks, b) integrate generative ai, agentic ai, claude code, codex, etc., c) share and continue progress at any point with other robots in its category, and d) collaborate on multi-site clinical trials (each derived from kevinkawchak/physical-ai-oncology-trials/tree/main/unification).

It is important that you clone the most current version of physical-ai-oncology-trials, and that your decisions be based on the current repository and reputable sources of information firstly from other reputable GitHub accounts and secondly from other reputable sources of online information (peer or non-peer reviewed). It is also important to cite other TRL frameworks as an influence such as the following, but the scope is much different in this project versus 1) ai-infrastructure-alliance/mltrl, 2) http://www.artemisinnovation.com/images/TRL_White_Paper_2004-Edited.pdf, 3) 10.1109/PICMET.2015.7273196. Mark the following paper as an inspiration due to recommending LLM usage for upcoming oncology trials: 4) 10.5281/ZENODO.17451709. Cite other sources you use properly.

Your main objective in this prompt is to start with 3 state-of-the-art open source robotic arms (multi-manufacturer preference) in the Collaborative Robots (Cobots) category each receiving their own USL rating based on a)-d) from above. The Readme for main/unification/usl should provide 3 different effective text diagrams illustrating general differences between each Cobot, technical differences between each Cobot, and scoring differences between them. Each Cobot model should have its own directory with useful and comprehensive code to aid its future unification process with other Cobot models from both inside their organization and outside their organization.

Be sure to fix and address errors that would cause failed checks for the single pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Provide an updated changelog (v1.4.0). Put this prompt into prompts.md under main. Place the new release notes in a releases.md under main using the format below. Update other relevant documentation such as project structures.

"FORMAT"
Release title
v1.4.0 -

## Summary

## Features

## Contributors
@kevinkawchak
@claude

## Notes
