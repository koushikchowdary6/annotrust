# Security

Annotrust reads annotation files you point it at and writes a report next to them. It does not open a network connection, and it does not phone home.

## What to report

- A report (especially the HTML file) that executes or loads remote content when opened from a local path.
- A parser bug that writes outside the output path you passed, or that can be aimed at a file you did not name.
- A crash on ordinary annotation input that a CI job would treat as a pass.

Label errors, missed attackers, and false accusations are quality bugs. Open a normal issue for those.

## How to report

Prefer a private GitHub security advisory on this repository if that feature is enabled. Otherwise email Sai Pranay Kowshik Chowdary Sakhamuri at pranaykoushikchowdary@gmail.com. Please include the version, the command line, and a minimized file that triggers the bug. Do not send a production label file if the labels themselves are sensitive; a synthetic file with the same shape is enough.

## Scope

The HTML report escapes text it interpolates. It is still a file generated from the annotation export. Treat that export as data, not as a program. The Encord and Label Studio readers are unofficial parsers of public export shapes. They are not a boundary those products provide.
