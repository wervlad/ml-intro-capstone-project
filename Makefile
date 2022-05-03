SOURCES = $(wildcard *.py) $(wildcard */*.py) $(wildcard */*/*.py)

profiling_report:
	poetry run profiling_report

tags:
	ctags -f tags -R --fields=+iaS --extra=+q $(SOURCES)

include_tags:
	ctags -f include_tags -R --languages=python --fields=+iaS --extra=+q \
		/usr/lib/python3.9/

sync_with_git:
	git fetch
	git reset origin/main --hard

clean:
	rm -rf tags include_tags __pycache__ */__pycache__ */*/__pycache__

.PHONY: clean include_tags tags

