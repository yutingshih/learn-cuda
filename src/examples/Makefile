all:

%:
	@for dir in *; do \
		if [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C $$dir $@; \
		fi; \
	done
