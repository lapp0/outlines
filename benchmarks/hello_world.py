# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_values(self):
        for value in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            d[key]


class FactorialSuite:
    def time_prime(self):
        def nth_prime(n):
            if n < 1:
                return None

            primes = []
            num = 2  # The first prime number
            while len(primes) < n:
                is_prime = True
                for prime in primes:
                    if num % prime == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(num)
                num += 1
            return primes[-1]

        nth_prime(10000)


class MemSuite:
    def mem_list(self):
        return [0] * 256
