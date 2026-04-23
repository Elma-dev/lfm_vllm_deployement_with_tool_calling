import modal
print(dir(modal))
if hasattr(modal, "experimental"):
    print("modal.experimental exists")
    print(dir(modal.experimental))
else:
    print("modal.experimental does not exist")
