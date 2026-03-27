/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * Tests closing a communicator while there are still inflight requests
 */

#include "config.h"
#include "functional_test.h"
#include <pthread.h>

class InflightCloseTest : public TestScenario {
public:
	explicit InflightCloseTest(size_t num_threads = 0, size_t num_iterations = 1)
		: TestScenario("Inflight Close Test", num_threads, num_iterations)
	{
		if (num_threads > 0) {
			pthread_barrier_init(&close_barrier, nullptr, num_threads);
		}
	}

	~InflightCloseTest() override {
		if (threads.size() > 0) {
			pthread_barrier_destroy(&close_barrier);
		}
	}

	void setup(ThreadContext& ctx) override {

		// First iteration: setup all connections via base class
		if (ctx.lcomms.empty()) {
			return TestScenario::setup(ctx);
		}

		// Subsequent iterations: only re-establish connections that were closed
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			if (ctx.lcomms[dev_idx] == nullptr) {
				ctx.setup_connection(dev_idx, 2);
			}
		}
	}

	void run(ThreadContext& ctx) override {
		auto gdr_support = get_support_gdr(ext_net);

		/* Per-device arrays for buffers and handles */
		std::vector<std::vector<void*>> all_buffers(ctx.lcomms.size());
		std::vector<std::vector<void*>> all_mhandles(ctx.lcomms.size());

		/* Phase 1: Post operations on all devices */
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			int physical_dev = ctx.device_map[dev_idx];
			int buffer_type = gdr_support[physical_dev] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
			auto &buffers = all_buffers[dev_idx];
			auto &mhandles = all_mhandles[dev_idx];
			buffers.resize(REQS_PER_DEV, nullptr);
			mhandles.resize(REQS_PER_DEV, nullptr);

			if (ctx.rank == 0) {
				for (int i = 0; i < REQS_PER_DEV; i++) {
					OFINCCLTHROW(allocate_buff(&buffers[i], DATA_SIZE, buffer_type));
					OFINCCLTHROW(initialize_buff(buffers[i], DATA_SIZE, buffer_type));
					OFINCCLTHROW(ext_net->regMr(ctx.scomms[dev_idx], buffers[i], DATA_SIZE, buffer_type, &mhandles[i]));
					void *req = nullptr;
					post_send(ext_net, ctx.scomms[dev_idx], buffers[i], DATA_SIZE, TAG, mhandles[i], &req);
				}
			} else {
				for (int i = 0; i < REQS_PER_DEV; i++) {
					OFINCCLTHROW(allocate_buff(&buffers[i], DATA_SIZE, buffer_type));
					OFINCCLTHROW(ext_net->regMr(ctx.rcomms[dev_idx], buffers[i], DATA_SIZE, buffer_type, &mhandles[i]));
					void *recv_bufs[] = {buffers[i]};
					size_t sizes[] = {DATA_SIZE};
					int tags[] = {TAG};
					void *handles[] = {mhandles[i]};
					void *req = nullptr;
					post_recv(ext_net, ctx.rcomms[dev_idx], 1, recv_bufs, sizes, tags, handles, &req);
				}
			}
		}

		/* Phase 2: Deregister memory on all devices */
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			for (int i = 0; i < REQS_PER_DEV; i++) {
				if (ctx.rank == 0) {
					OFINCCLTHROW(ext_net->deregMr(ctx.scomms[dev_idx], all_mhandles[dev_idx][i]));
				} else {
					OFINCCLTHROW(ext_net->deregMr(ctx.rcomms[dev_idx], all_mhandles[dev_idx][i]));
				}
			}
		}

		/* Barrier: ensure all threads finished posting before any thread
		   starts closing. fi_close in the EFA provider touches shared
		   domain state, so concurrent close + post causes corruption. */
		if (threads.size() > 0) {
			pthread_barrier_wait(&close_barrier);
		}

		/* Phase 3: Close all communicators (endpoint abort happens here) */
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			OFINCCLTHROW(ext_net->closeSend(ctx.scomms[dev_idx]));
			ctx.scomms[dev_idx] = nullptr;
			OFINCCLTHROW(ext_net->closeRecv(ctx.rcomms[dev_idx]));
			ctx.rcomms[dev_idx] = nullptr;
			OFINCCLTHROW(ext_net->closeListen(ctx.lcomms[dev_idx]));
			ctx.lcomms[dev_idx] = nullptr;
		}

		/* Phase 4: Cleanup buffers */
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			int physical_dev = ctx.device_map[dev_idx];
			int buffer_type = gdr_support[physical_dev] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
			for (int i = 0; i < REQS_PER_DEV; i++) {
				if (all_buffers[dev_idx][i]) {
					OFINCCLTHROW(deallocate_buffer(all_buffers[dev_idx][i], buffer_type));
				}
			}
		}
	}

	void teardown(ThreadContext& ctx) override {
		return TestScenario::teardown(ctx);
	}

private:
	static constexpr size_t DATA_SIZE = 1024 * 1024;
	static constexpr int TAG = 1;
	static constexpr int REQS_PER_DEV = 8;
	pthread_barrier_t close_barrier;
};

int main(int argc, char* argv[])
{
	TestSuite suite;
	InflightCloseTest test(0, 10);      // single-threaded, 10 iterations
	InflightCloseTest mt_test(4, 10);   // 4 threads, 10 iterations each
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}
