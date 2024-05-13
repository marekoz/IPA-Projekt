#include <stdio.h>
#include <inttypes.h>
#include <x86intrin.h>


class InstructionCounter
{
private:
	unsigned  long long   counter;
	unsigned  long long  end_count;
	unsigned  long long total_count;
public:
	
	void start()
	{
		counter= __rdtsc();
	};

	void end()
	{
		end_count = __rdtsc();
	};

	void print()
	{
		printf("%I64lld \n", getCyclesCount());
	}

	unsigned long long  getCyclesCount()
	{
		end();
		total_count = end_count - counter;
		return  total_count;
	}
};
