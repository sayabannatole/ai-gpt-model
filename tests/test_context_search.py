import unittest

from src.context_search import MiniContextModel, PageDocument


class MiniContextModelTests(unittest.TestCase):
    def test_search_returns_most_relevant_document_first(self):
        model = MiniContextModel()
        model.fit(
            [
                PageDocument(url="/pricing", title="Pricing", content="Plans and price tiers"),
                PageDocument(url="/docs", title="Documentation", content="API guides and SDK setup"),
            ]
        )

        results = model.search("api setup", top_k=1)
        self.assertEqual(results[0][0].url, "/docs")
        self.assertGreater(results[0][1], 0)

    def test_serialization_round_trip(self):
        model = MiniContextModel()
        model.fit([PageDocument(url="/", title="Home", content="welcome home page")])

        payload = model.to_dict()
        reloaded = MiniContextModel.from_dict(payload)

        results = reloaded.search("welcome", top_k=1)
        self.assertEqual(results[0][0].title, "Home")


if __name__ == "__main__":
    unittest.main()
