"""Public Fire Explorer: the science-museum-facing experience.

A presentation + interaction layer over the *same* FDS data the research
application reads -- ScenarioStore, TimeController, the cinema/ rendering
pipeline, and events.py's detected story beats are all reused as-is. This
package computes no physics of its own; every number it shows a visitor
came out of the store or an existing analysis module.

Kept as a separate package (not a pages/ entry) because public mode is a
root-level experience that replaces the researcher shell entirely -- see
public/experience.py for why.
"""
