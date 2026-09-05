"""Static pickletools inspection of an old local tuning artifact.

Never unpickles or executes the artifact; only disassembles opcodes and records
referenced module/class names and string constants.
"""
from pathlib import Path
import hashlib, json, pickletools, re

P=Path(r"C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904\tune_local.pkl")
raw=P.read_bytes(); ops=list(pickletools.genops(raw))
globals_=[]; strings=[]; ints=[]
for op,arg,pos in ops:
    if op.name in {"GLOBAL","STACK_GLOBAL","INST","OBJ","NEWOBJ","NEWOBJ_EX"}: globals_.append({"opcode":op.name,"arg":repr(arg),"pos":pos})
    if op.name in {"BINUNICODE","SHORT_BINUNICODE","UNICODE"} and isinstance(arg,str): strings.append(arg)
    if op.name in {"BININT","BININT1","BININT2","LONG_BINPUT"} and isinstance(arg,int): ints.append(arg)
meta={"path":str(P),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"opcode_count":len(ops),"globals":globals_,"strings_unique":sorted(set(strings)),"ints_sample":ints[:300]}
Path(__file__).with_name("tune_local_pickle_static_20260905.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in meta.items() if k not in {"globals","strings_unique","ints_sample"}},indent=2))
print("globals",globals_)
print("strings",sorted(set(strings)))
