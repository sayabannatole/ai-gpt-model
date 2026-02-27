import unittest

from src.context_search import MiniContextModel, PageDocument, SectionExtractor, infer_tags


class MiniContextModelTests(unittest.TestCase):
    def test_search_returns_most_relevant_document_first(self):
        model = MiniContextModel()
        model.fit(
            [
                PageDocument(
                    url="/pricing",
                    title="Pricing",
                    section="Plans",
                    content="Plans and price tiers",
                    tags=["docs"],
                ),
                PageDocument(
                    url="/docs",
                    title="Documentation",
                    section="API Setup",
                    content="API guides and SDK setup",
                    tags=["docs"],
                ),
            ]
        )

        results = model.search("api setup", top_k=1)
        self.assertEqual(results[0][0].url, "/docs")
        self.assertGreater(results[0][1], 0)

    def test_search_can_filter_by_tags(self):
        model = MiniContextModel()
        model.fit(
            [
                PageDocument(
                    url="/docs/getting-started",
                    title="Documentation",
                    section="Install",
                    content="Install the SDK",
                    tags=["docs"],
                ),
                PageDocument(
                    url="/blog/release",
                    title="Release blog",
                    section="Announcement",
                    content="Product release and launch",
                    tags=["blog"],
                ),
            ]
        )

        results = model.search("release", top_k=5, tags=["docs"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].url, "/docs/getting-started")

    def test_serialization_round_trip(self):
        model = MiniContextModel()
        model.fit(
            [
                PageDocument(
                    url="/",
                    title="Home",
                    section="Overview",
                    content="welcome home page",
                    tags=["docs"],
                )
            ]
        )

        payload = model.to_dict()
        reloaded = MiniContextModel.from_dict(payload)

        results = reloaded.search("welcome", top_k=1)
        self.assertEqual(results[0][0].title, "Home")


class SectionParsingTests(unittest.TestCase):
    def test_section_extractor_splits_content_by_headings(self):
        parser = SectionExtractor()
        parser.feed(
            """
            <html><body>
              <h1>Intro</h1>
              <p>Hello world</p>
              <h2>Usage</h2>
              <p>Run the tool</p>
            </body></html>
            """
        )

        self.assertEqual(parser.sections, [("Intro", "Hello world"), ("Usage", "Run the tool")])

    def test_infer_tags_supports_expected_metadata_tags(self):
        self.assertEqual(infer_tags("https://example.com/docs/start"), ["docs"])
        self.assertEqual(infer_tags("https://example.com/blog/update"), ["blog"])
        self.assertEqual(infer_tags("https://example.com/changelog/v2"), ["changelog"])


if __name__ == "__main__":
    unittest.main()
