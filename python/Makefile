.PHONY: build-base-image test-pylightnet

# Build the base Docker image from the repository root
build-base-image:
	cd .. && docker build -f python/Dockerfile -t trt-lightnet-base .

# Test that pylightnet can be installed properly in the container
test-pylightnet: build-base-image
	docker run --rm trt-lightnet-base /bin/bash -c " \
		python3 -m pip install --upgrade setuptools && \
		cd python && \
		python3 -m pip install --no-cache-dir .[dev] && \
		cd .. && \
		python3 -m pytest python/tests/ \
	"