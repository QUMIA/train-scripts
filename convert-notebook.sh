#!/bin/bash


# exclude cells with tag 'exclude'
# exclude markdown cells

jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['exclude']" \
    --TemplateExporter.exclude_markdown=True \
    --to python $1


# doens't work:  --RegexRemovePreprocessor.patterns="^%"
