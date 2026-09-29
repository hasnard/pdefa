# pdefa - AGPL-3.0-or-later - see LICENSE
"""Custom model example — feed your own tensor.

Usage:
    python 06_custom_model.py <model.engine>
"""
import sys
import pdefa

if len(sys.argv) < 2:
    print("usage: python 06_custom_model.py <model.engine>")
    sys.exit(1)

ENGINE = sys.argv[1]

ctx = pdefa.Context()
model = pdefa.Model(ctx, ENGINE)
session = pdefa.Session(ctx, model)

info = model.input_info(0)
print(f"Input required: {info.name} {info.shape}")

shape = list(info.shape)
tensor = pdefa.zeros(*shape)

session.set_input(0, tensor)
session.run()

raw = session.output(0).tolist()
print(f"Output shape: {session.output(0).shape}")
print(f"First values: {raw[0][:4] if raw and raw[0] else 'empty'}")