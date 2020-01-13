import ukf 
import unittest
import numpy as np

class TestUkf(unittest.TestCase):

    def test_dist(self):
        pt1 = [0,0]; pt2 = [5,5]
        sigma_range = 0; sigma_bearing = 0
        dist_formula = np.sqrt( (pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 )
        result = ukf.dist2(pt1,pt2,sigma_range,sigma_bearing)
        self.assertEqual(result,dist_formula)
        
        



if __name__ =='__main__':
    unittest.main()
