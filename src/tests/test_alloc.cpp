#include <luisa/luisa-compute.h>
#include <vector>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

namespace dummy {

template<typename T>
struct LuisaExternalBuffer {
	static_assert(luisa::compute::is_valid_buffer_element_v<T>);
	bool is_located = false;
	luisa::compute::Buffer<T> buf;
	T* ptr = nullptr;
	size_t N = 0;
	void locate(luisa::compute::Device& device) {
		if (!is_located) {
			buf = device.import_external_buffer<T>(ptr, N);
			is_located = true;
		}
	}
};

template<typename T>
void soa_obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment) {
	std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);// offset = minimum k * alignment > chunk
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}

template<typename T>
size_t soa_required(size_t N) {
	char* size = nullptr;
	T::from_chunk(size, N);
	return ((size_t)size) + 128;
}

template<typename T>
size_t inno_soa_required(size_t N) {
	return soa_required<T>(N);
}

template<typename T>
void inno_soa_obtain(char*& chunk, LuisaExternalBuffer<T>& buf, size_t N) {
	soa_obtain(chunk, buf.ptr, N, 128);
	buf.N = N;
}


struct AllocDummy {
	LuisaExternalBuffer<int> d_int;
	LuisaExternalBuffer<uint> d_uint;
	static AllocDummy from_chunk(char*& chunk, size_t N) {
		AllocDummy dummy;
		inno_soa_obtain(chunk, dummy.d_int, N);
		inno_soa_obtain(chunk, dummy.d_uint, N);
		return dummy;
	}
	void locate(Device& device) {
		d_int.locate(device);
		d_uint.locate(device);
	}
};// struct AllocDummy;

std::function<char*(size_t N)> luisa_alloc(Device& device, Buffer<uint>& buf) {
	// input size in bytes
	// output buffer
	// make sure the input size be aligned to sizeof(uint)
	auto lambda = [&device, &buf](size_t N) {
		auto N_in_uint = N / sizeof(uint);
		LUISA_INFO("N_in_uint: {}", N_in_uint);
		buf = device.create_buffer<uint>(N_in_uint);
		char* native_handle = static_cast<char*>(buf.native_handle());
		return native_handle;
	};
	return lambda;
}

}


int main(int argc, char *argv[]) {
    using namespace dummy;
    luisa::log_level_verbose();
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
	constexpr int N = 100;
	std::vector<int> h_int(N);
	std::vector<uint> h_uint(N);
	// init 1 ~ N with iota
	std::iota(h_int.begin(), h_int.end(), 1);
	std::iota(h_uint.begin(), h_uint.end(), 1);
	// allocate device memory
	auto chunk_size = inno_soa_required<AllocDummy>(N);
    LUISA_INFO("chunk_size: {}", chunk_size);// 0 -> 0 -> 400 -> 512 -> 912 + 128 = 1040
	// allocator begin
	Buffer<uint> dummy_buf;
	auto dummy_buffer = luisa_alloc(device, dummy_buf);
	// char* chunk = dummy_buffer(N); // FUCK !!!!!
	char* chunk = dummy_buffer(chunk_size);
	char* ptr = chunk;
	// allocator end

	AllocDummy dummy = AllocDummy::from_chunk(ptr, N);
	dummy.locate(device);

	auto stream = device.create_stream();
    stream << dummy.d_int.buf.copy_from(h_int.data());
	stream << dummy.d_uint.buf.copy_from(h_uint.data());

	Kernel1D<Buffer<int>> kernel_init_int = [&](BufferVar<int> buffer) {
		auto i = thread_id().x;
		buffer.write(i, Int(i));
	};
	Kernel1D<Buffer<uint>> kernel_init_uint = [&](BufferVar<uint> buffer) {
		auto i = thread_id().x;
		buffer.write(i, UInt(i));
	};

	auto shader_init_int = device.compile(kernel_init_int);
	auto shader_init_uint = device.compile(kernel_init_uint);

	stream << shader_init_int(dummy.d_int.buf).dispatch(N) << shader_init_uint(dummy.d_uint.buf).dispatch(N);

	stream << synchronize();


	Kernel1D<Buffer<int>> kernel_def_int = [&](BufferVar<int> buffer) {
		auto i = thread_id().x;
		buffer.write(i, buffer.read(i) + 1);
	};
	Kernel1D<Buffer<uint>> kernel_def_uint = [&](BufferVar<uint> buffer) {
		auto i = thread_id().x;
		buffer.write(i, buffer.read(i) + 1);
	};

	auto shader_int = device.compile(kernel_def_int);
	auto shader_uint = device.compile(kernel_def_uint);

	stream << shader_int(dummy.d_int.buf).dispatch(N) << shader_uint(dummy.d_uint.buf).dispatch(N) << synchronize();

	std::vector<int> result_int(N);
	std::vector<uint> result_uint(N);

	stream << dummy.d_int.buf.copy_to(result_int.data());
	stream << dummy.d_uint.buf.copy_to(result_uint.data());
	stream << synchronize();

	for (int i = 0; i < 10; i++) {
        LUISA_INFO("result_int[{}]: {}, result_uint[{}]: {}", i, result_int[i], i, result_uint[i]); // should be [i] -> i + 1
		// CHECK(result_int[i] == h_int[i] + 1);
		// CHECK(result_uint[i] == h_uint[i] + 1);
	}

	ptr = chunk;
	AllocDummy dummy2 = AllocDummy::from_chunk(ptr, N);
	dummy2.locate(device);

	// now dummy2 data is changed too, we add 1 once more
	stream << shader_int(dummy2.d_int.buf).dispatch(N) << shader_uint(dummy2.d_uint.buf).dispatch(N) << synchronize();
	stream << dummy2.d_int.buf.copy_to(result_int.data()) << dummy2.d_uint.buf.copy_to(result_uint.data()) << synchronize();

	for (int i = 0; i < N; i++) {
        LUISA_INFO("result_int[{}]: {}, result_uint[{}]: {}", i, result_int[i], i, result_uint[i]); // should be [i] -> i + 1
		// CHECK(result_int[i] == h_int[i] + 2);
		// CHECK(result_uint[i] == h_uint[i] + 2);
	}
	return 0;
}
