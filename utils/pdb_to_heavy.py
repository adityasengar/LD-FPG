import MDAnalysis as mda

u = mda.Universe("prot.pdb")
# Selection for "all heavy atoms" can be:
heavy = u.select_atoms("not (name H* or element H)")
heavy.write("heavy_only.pdb")
