include ../build_tools/poetry.mk

test:
	pytest tests/ -vv -s
