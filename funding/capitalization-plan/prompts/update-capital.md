## update-capital

For physical-ai-oncology-trials/tree/claude/chemicalqdevice-conversion-paper-pc6yaz/funding/capitalization-plan: update files based on the following instructions. (Also update this directory’s README with your new changes.)

* Make sure final-capital-LaTeX.zip and main.pdf are fully correct and receive their last updates at the same time.
* For each diagram: start each caption text with “Figure 1. “, “Figure 2 . “… until the last figure.
* Do the same with each table caption text beginning with “Table 1. “, “Table 2. “, until the last table.
* Make sure the new captions for both figures and tables are a) centered, b) have a similar number of characters per caption line of text, and c) have up to three lines of caption text.
* Make sure other figure and table reference tables and other paper mentions match your updated information (ie. number, section, directory (if present), and question headings).
* Be certain that each of the figures and tables are also referenced once each in the body’s text (ie: “as illustrated in Figure 6”, or “as depicted in Table 11”, or other custom reference phrases).


Add a new update-capital.md under physical-ai-oncology-trials/tree/claude/chemicalqdevice-conversion-paper-pc6yaz/funding/capitalization-plan/prompts that adapts to the prompt-capital.md format with this current prompt information.

It is also important to ensure that figures, tables, and caption text are all properly centered in the page in the x direction. If this is an issue: adapt using the following code.
\usepackage{changepage}
START DIAGRAM X SHIFT
\newlength{\DiagramXShift}
\setlength{\DiagramXShift}{ mm}
\begin{appfloat}[!tb]
\begin{adjustwidth}
  {\DiagramXShift}
  {\dimexpr-\DiagramXShift\relax}
\begin{appfig}[plantuml-type / use case]
\end{appfig}
\end{adjustwidth}

* Currently, the DOI links and other URLs need to be clickable in the reference section. There is only text now. Fix the respective .sty to finish this task.


* Use term "projected" instead of "estimated" regarding the $36,330 valuation.


Make sure there will be no repository notification errors, and that the updated PR will pass all GitHub checks on the merge by the user.
