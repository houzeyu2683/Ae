import sparrow
import inspect
import os
import shutil
import base64

sparrow =  sparrow.Sparrow(device='cuda')
module = inspect.getmodule(sparrow)
with open(inspect.getfile(module), 'r') as paper:
    content = paper.read()
    pass
_ = paper
code = base64.b64encode(
    content.encode("utf-8")
)
path = './model.bin'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "wb") as paper:
    paper.write(code)
    pass
_ = paper



with open(path, "rb") as paper:
    code = paper.read()
    pass
_ = paper
content = base64.b64decode(code).decode("utf-8")
print(content)

exec(content)

