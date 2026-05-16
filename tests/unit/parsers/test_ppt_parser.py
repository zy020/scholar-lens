import pytest
from scholar_lens.parsers.ppt_parser import PPTParser


class TestPPTParser:
    def test_instantiation(self):
        parser = PPTParser()
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_parse_nonexistent_raises(self):
        parser = PPTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.pptx")
