"""
Finding your way around the bank
================================

Thirty-odd methods is too many to remember and too few to need a search engine. Four
calls cover it: ``categories`` for the taxonomy, ``list_methods`` for the table,
``available`` for the names, ``describe`` for one method in full.
"""
import attnbank as ab

print("THE TAXONOMY -- what each category changes in y(q) = int v dpi_q\n")
for name, what in ab.categories().items():
    print(f"  {name:<13}{what}")

print("\n\nTHE TABLE -- grouped, with the declaration columns that drive the test suite")
print(ab.list_methods("factorize"))

print("\n\nFILTERS -- every question about the bank is one call")
print("  causal + row-stochastic:", ab.list_methods(causal=True).filter(
    row_stochastic=True).names)
print("  a different operator:   ", ab.available(objective="modify"))
print("  cannot express weights: ", [n for n in ab.available()
                                     if not ab.get_class(n).expresses_weights])

print("\n\nONE METHOD IN FULL\n")
print(ab.describe("nystrom"))

print("\n\nAND THE CONTRACT, ON DEMAND\n")
print(ab.check_conformance("nystrom", r=8))
