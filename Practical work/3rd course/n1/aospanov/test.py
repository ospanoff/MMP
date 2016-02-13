import unittest
import numpy as np
import num1
import num2
import num3
import num4
import num6
import num7
import num8

# 1st task testing class
class Test_num1(unittest.TestCase):

    # setting up initial values
    def setUp(self):
        n = int(np.random.rand() * 10 + 3)
        m = int(np.random.rand() * 10 + 3)
        self.val = np.random.rand(n, m)

    # testing of equivalent of 1st and 2nd methods
    def test_first_eq_second(self):
        self.assertEqual(
            num1.first_method(self.val), num1.second_method(self.val))

    # testing of equivalent of 2st and 3nd methods
    def test_second_eq_third(self):
        self.assertEqual(
            num1.first_method(self.val), num1.third_method(self.val))

    # testing of equivalent of 3st and 4nd methods
    def test_third_eq_fourth(self):
        self.assertEqual(
            num1.third_method(self.val), num1.fourth_method(self.val))

# following the same

class Test_num2(unittest.TestCase):

    n = int(np.random.rand() * 10) + 3

    a = np.random.randint(0, n, size=n)
    b = np.random.randint(0, n, size=n)

    def setUp(self):
        n = int(np.random.rand() * 10) + 3
        a = np.random.randint(0, n, size=n)
        b = np.random.randint(0, n, size=n)
        self.val = (np.random.rand(self.n, self.n) * 10).astype(np.int32)

    def test_first_eq_second(self):
        self.assertTrue(np.all(np.array(
                        num2.first_method(self.val, self.a, self.b) ==
                        num2.second_method(self.val, self.a, self.b))))

    def test_second_eq_third(self):
        self.assertTrue(np.all(np.array(
                        num2.second_method(self.val, self.a, self.b) ==
                        num2.third_method(self.val, self.a, self.b))))


class Test_num3(unittest.TestCase):

    n = int(np.random.rand() * 10 + 3)
    val1 = np.random.rand(n)
    val2 = np.random.rand(n)

    def setUp(self):
        n = int(np.random.rand() * 10 + 3)
        self.val1 = np.random.rand(self.n)
        self.val2 = np.random.rand(self.n)

    def test_first_eq_second(self):
        self.assertEqual(
            num3.first_method(self.val1, self.val2),
            num3.second_method(self.val1, self.val2))

    def test_second_eq_third(self):
        self.assertEqual(
            num3.first_method(self.val1, self.val2),
            num3.third_method(self.val1, self.val2))


class Test_num4(unittest.TestCase):

    n = int(np.random.rand() * 10) + 3
    i = np.random.randint(0, n - 1, size=2)

    def setUp(self):
        n = int(np.random.rand() * 10) + 3
        i = np.random.randint(0, n - 1, size=2)
        self.val = np.random.rand(n) * 10
        self.val[i] = 0

    def test_first_eq_second(self):
        self.assertEqual(
            num4.first_method(self.val), num4.second_method(self.val))

    def test_second_eq_third(self):
        self.assertEqual(
            num4.first_method(self.val), num4.third_method(self.val))


class Test_num6(unittest.TestCase):

    def setUp(self):
        n = int(np.random.rand() * 10 + 3)
        self.val = np.random.randint(0, n, size=n)

    def test_first_eq_second(self):
        self.assertTrue(np.all(
                        np.array(num6.first_method(self.val)[0] ==
                                 num6.second_method(self.val)[0]) ==
                        np.array(num6.first_method(self.val)[1] ==
                                 num6.second_method(self.val)[1])))

    def test_second_eq_third(self):
        self.assertTrue(np.all(
                        np.array(num6.third_method(self.val)[0] ==
                                 num6.second_method(self.val)[0]) ==
                        np.array(num6.third_method(self.val)[1] ==
                                 num6.second_method(self.val)[1])))


class Test_num7(unittest.TestCase):

    n = int(np.random.rand() * 10 + 3)
    m = int(np.random.rand() * 10 + 3)
    k = int(np.random.rand() * 10 + 3)
    val1 = np.random.rand(n, k)
    val2 = np.random.rand(m, k)

    def setUp(self):
        n = int(np.random.rand() * 10 + 3)
        m = int(np.random.rand() * 10 + 3)
        k = int(np.random.rand() * 10 + 3)
        self.val1 = np.random.rand(n, k)
        self.val2 = np.random.rand(m, k)

    def test_first_eq_second(self):
        self.assertTrue(np.all(np.array(
            num7.first_method(self.val1, self.val2) -
            num7.second_method(self.val1, self.val2) < 0.001)))

    def test_second_eq_third(self):
        self.assertTrue(np.all(np.array(
            num7.second_method(self.val1, self.val2) -
            num7.third_method(self.val1, self.val2) < 0.001)))


class Test_num8(unittest.TestCase):
    N = int(np.random.rand() * 10 + 3)
    D = int(np.random.rand() * 10 + 3)
    val = (np.random.rand(N, D) * 10).astype(np.int32)
    m = np.random.randint(1, D * N, size=D)
    C = np.diagflat(m * np.random.randint(1, N * D, size=1))

    def setUp(self):
        N = int(np.random.rand() * 10 + 3)
        D = int(np.random.rand() * 10 + 3)
        self.val = (np.random.rand(N, D) * 10).astype(np.int32)
        self.m = np.random.randint(1, D * N, size=D)
        self.C = np.diagflat(self.m * np.random.randint(1, N * D, size=1))

    def test_first_eq_second(self):
        self.assertTrue(np.all(np.array(
            num8.first_method(self.val, self.m, self.C) -
            num8.second_method(self.val, self.m, self.C) < 0.001)))

    def test_second_eq_third(self):
        self.assertTrue(np.all(np.array(
            num8.second_method(self.val, self.m, self.C) -
            num8.third_method(self.val, self.m, self.C) < 0.001)))


if __name__ == '__main__':
    unittest.main()
