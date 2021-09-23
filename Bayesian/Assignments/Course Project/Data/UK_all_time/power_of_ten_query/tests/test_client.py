import unittest
from po10.client import Client



class TestAthleteMethods(unittest.TestCase):

    def test_get_athlete(self):
        SEB_COE_ID = 1987
        seb = Client().get_athlete(SEB_COE_ID)
        self.assertEqual(seb.id, SEB_COE_ID)
        self.assertEqual(seb.nation, "England")
        self.assertEqual(seb.name, "Sebastian Coe")