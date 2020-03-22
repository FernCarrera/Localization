import study 
import unittest
import numpy as np

class TestStudy(unittest.TestCase):

    def test_normalize_angle(self):
        test_num = 3*np.pi
        value = study.normalize_angle(test_num)  # about 370deg
        
        #while test_num > np.pi:
        #    test_num -= 2.0*np.pi

        test_num = test_num % (2.0*np.pi)
        if (test_num<0):
            test_num+= (2.0*np.pi)
        print(test_num)
        self.assertEqual(value,test_num)





if __name__ == '__main__':
    unittest.main()
