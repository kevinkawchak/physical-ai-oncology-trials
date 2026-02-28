# Development Prompts — Patient-Robot Instructions

Archive of prompts used to develop the patient-robot instructional illustrations.

---

## v1.9.0 — Patient-Robot Instructions: Physical AI Oncology Trials (February 28, 2026)

Based on kevinkawchak/physical-ai-oncology-trials, your goal is to create patient-facing instructional illustrations in a high resolution and clear LaTeX formatted portrait pdf with 10 pages. Each page should have their own unique professional and prominent black and white portrait layout image (black draw, white background). Each page should have a consistent feel in terms of image artistry and size, text formatting, title placement, etc. The purpose of the pdf is for an upcoming patient to visualize, read, and feel comfortable regarding how to correctly interact with a specific type of robot for their upcoming physical ai oncology trial (that patient should have all the information they need on that single page for that robot type)(only 1 human patient and 1 robot per page, no human doctors or nurses). Patient diversity should be apparent (even for the black and white images). 2 of the images (starting after the forth image) should be children of suitable age for a pediatric oncology trial, and matched to robots suitable to their size and most likely to be used on children).

Based on the list of 13 below (Must use Cobots, Surgical robots, Humanoids): pick the Top 10 based on which are most appropriate to physical ai oncology trials, (the image order should based on which would be used most commonly, starting with the most common on page 1. Store all files under kevinkawchak/physical-ai-oncology-trials/patients and its corresponding subdirectories (include a detailed Readme describing the paper and files under /patients). In addition to 1 image per page in the pdf, also save each image as its own high resolution and clear versions of pdf, svg, and png in folders based on format type (ie svg directory for svg images).

List of 13:
Cobots, Surgical robots, Humanoids, Telepresence robots, Social companion robots, Autonomous hospital transport robots (AMRs), Radiotherapy patient-positioning robots, Robotic needle-placement systems, Steerable needle / needle-steering robots, Imaging assistant robots, Radiotherapy motion-management / tracking robots, UV disinfection robots, Rehabilitation exoskeletons / robotic gait trainers

Consider ISO 15223-1, ISO 20417, ISO 7000, IEC 60417 where appropriate for symbols. Consider ISO 7010 and ISO 3864-1 where appropriate for safety pictograms.

a) Above the top bar for each page should be suitably small text containing the following:
Kevin Kawchak, CEO ChemicalQDevice, https://orcid.org/0009-0007-5457-8667, kevink@chemicalqdevice.com.

b) The title below the top bar is "Patient-Robot Instructions: Physical AI Oncology Trials", and should be suitably large. For each page, append the title with the robot type shown in its image, such as "- Cobots" matching the capitalization of the base title.

c) The prominent image for each robot and human patient on each page should be professional and descriptive enough to recognize the differences between robot types, but not overly complex to avoid patient confusion. Make sure illustrations of each robot type are well researched so that the robot would be universally recognized by many as that specific type. Avoid robot logos and too many features that would indicate a specific manufacturer. Make sure that the adult or child patient who have cancer are interacting with the robot in the most likely and appropriate ways typical in upcoming physical ai oncology trials.

d) Below the image use bullet points and numbered lists where appropriate and other types of formatted text to avoid long paragraphs (the context is that the patient has cancer and needs to interact with robots regarding medical activities) Full instructions are needed to prepare each patient. Use appropriate instructions regarding 1) any patient preparations at home, and especially 2) What the patient needs to do the moment the patient walks into the room alone with the robot 3) What the patient needs to do while interacting with the robot, 4) What the patient needs to do in concluding the session with the robot, 5) What the patient needs to do at home and for any follow-ups with the robot. Use quantitative data like the estimated number of minutes each patient will interact with the robot, what exactly the patient needs to do with their hands at certain times, how to talk to the robot or engage in other forms of communication with the robot. This section should look consistent across the ten pages.

e) Below the bottom bar for each page should be suitably small text containing the following: today's date, 10.5281/zenodo.18810541 (https://doi.org/10.5281/zenodo.18810541), Claude Code Opus 4.6, and the current page number. Provide truncated links from online repositories and sources only on the specific page they relate to (.bib should have all links from all pages).

Make sure to clone the current repo and utilize appropriate information regarding this pdf. Search other GitHub repositories and online sites if additional context or robot visual assistance is needed. Make sure every page is properly formatted and seems attractive to read. Make sure there are not large gaps between words if using latex raggedright. Make sure information across pages is not exactly duplicate.

Be sure to fix and address errors that would cause failed checks for the single pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " Place the new release notes in releases.md under main using the format below (note title gets no hashes, while summary, features, etc. get two leading hashes). Update other relevant documentation such as project structures. Update the main Readme diagrams, repository structure, etc. where necessary. Provide an updated changelog (v1.9.0). Provide a copy of this prompt under a new kevinkawchak/physical-ai-oncology-trials/patients/prompts.md

Output the finished pdf paper with file name "Patient-Robot Instructions: Physical AI Oncology Trials" as a .pdf under /main/paper. Output a zip file containing 4 files titled "Latex Source Code" under /main/paper/: .tex,.sty, README, .bib. When you are finished, auto-push the update to GitHub on your own for my review. The user will then review your updates in GitHub prior to finalization.

"FORMAT"
Release title
v1.9.0 - [Fill in Title Here]

## Summary

## Features

## Contributors
@kevinkawchak
@claude

## Notes
