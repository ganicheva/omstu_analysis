import unittest
from main import tfidf

class TestTFIDF(unittest.TestCase):
    
    def test1(self):
        sent = ['заяц', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово']
        docs = [['заяц', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово', 'слово'],
                ['заяц'],
                ['заяц', 'слово'],
                ['слово', 'слово'],
                ['слово']]
        word = 'заяц'
        self.assertEqual(tfidf(word, sent, docs), 0.0222)
        
    def test2(self):
        sent = ['заяц', 'заяц', 'слово']
        docs = [['заяц', 'заяц', 'слово'],
                ['слово'],
                ['слово', 'слово']]
        word = 'заяц'
        self.assertEqual(tfidf(word, sent, docs), 0.3181)
        
    def test3(self):
        sent = ['заяц']
        docs = [['заяц']]
        word = 'заяц'
        self.assertEqual(tfidf(word, sent, docs), 0.0)
        
    def test4(self):
        sent = ['заяц']
        docs = [['заяц'],
                ['слово']]
        word = 'заяц'
        self.assertEqual(tfidf(word, sent, docs), 0.301)

if __name__ == "__main__":
  unittest.main()
