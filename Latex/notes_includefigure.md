# Instructions for latex to eps to png

### Step 1: Enable Syscall in LaTeX Workshop

If using vscode with the LaTeX Workshop extension, add the following recipe to your .vscode/settings.json:

To the latex-workshop.latex.tools list add:

```json
{
  "name": "pdflatex_syscall",
  "command": "pdflatex",
  "args": ["--shell-escape", "%DOC%.tex"]
}
```

And to `latex-workshop.latex.recipes` add the following recipe:

```json
{
  "name": "EPSBuild",
  "tools": ["pdflatex_syscall"]
}
```

### Step 2: Create your tikz image

Use the following template:

```latex
\documentclass{article}
\usepackage[paperwidth=7.16in, paperheight=4.0in, margin=0in]{geometry}

\usepackage{tikz}
\usetikzlibrary{external}
\tikzset{external/system call={pdflatex \tikzexternalcheckshellescape -halt-on-error
-interaction=batchmode -jobname "\image" "\texsource" && % or ;
pdftops -eps "\image".pdf}}
\tikzexternalize

\begin{document}
\begin{tikzpicture}
% Your Image Here
\end{tikzpicture}
\end{document}
```

### Step 3: Convert to png

Fire up Gimp and import the eps using the following settings:

- Resolution / Auflösung: 300 (dpi)
- Width / Breite: 2130 for 7.16 inch, as: 2130 = 300 \* 7.1
- Save as png

### Step 4: Include in the manuscipt
