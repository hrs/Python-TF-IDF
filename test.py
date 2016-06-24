#!/usr/bin/env python
import unittest
from tfidf import tfidf



class Test_tfidf(unittest.TestCase):
    def setUp(self):
        self.table = tfidf()
        self.table.addDocument("foo", ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"])
        self.table.addDocument("bar", ["alpha", "bravo", "charlie", "india", "juliet", "kilo"])
        self.table.addDocument("baz", ["kilo", "lima", "mike", "november"])


    def test_similarities(self):
        result = self.table.similarities(["alpha", "bravo", "charlie", "india"])

        for doc in result:
            name_doc, similarities = doc
            self.assertTrue(similarities <= 1)



if __name__ == "__main__":
	""" Runs the test class. """

	unittest.main()
