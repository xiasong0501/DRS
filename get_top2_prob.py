a = torch.rand(20, 1)
print(a)
cAHat = counts_selection.argmax().item()
print(cAHat)
a = a[torch.arange(a.size(0))!=cAHat]
cBHat = counts_selection.argmax().item()
print(cBHat)
print(a)

