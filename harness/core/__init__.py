"""core — pure-numpy shared primitives (no bpy, no mathutils).

Importable from EVERY environment: Blender's bundled python (rig/, views/) and
the artscript env (analysis/, tools/). The rule is that nothing under core/ may
import bpy or mathutils, so it stays a drop-in for the numpy-only envs. Mirrors the rig/lie.py precedent (pure numpy, cross-checked
against Blender in tests).
"""
