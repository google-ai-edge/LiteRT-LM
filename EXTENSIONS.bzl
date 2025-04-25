load("@bazel//tools/build_defs/repo:http.bzl", "http_archive")

def http_archives(ctx):
    """Lists http_archives not registered as bazel modules yet.

    Keep the alphabetic order. To get sha256 value, run:

    % curl -sL <url> | sha256sum -
    """

    use_repo_rule(
        http_archive,
        name = "minizip-ng",
        url = "https://github.com/zlib-ng/minizip-ng/archive/refs/tags/4.0.9.tar.gz",
        sha256 = "353a9e1c1170c069417c9633cc94ced74f826319d6d0b46d442c2cd7b00161c1",
        strip_prefix = "minizip-ng-4.0.9",
    )

    use_repo_rule(
        http_archive,
        name = "litert",
        url = "https://github.com/google-ai-edge/LiteRT/archive/refs/tags/v1.2.0.tar.gz",
        sha256 = "6ee80a0890f35de97b96d438bd99f2b1c852c81182a95a1059aacd7e2f5146ed",
        strip_prefix = "LiteRT-1.2.0",
    )

    use_repo_rule(
        http_archive,
        name = "sentencepiece",
        url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz",
        sha256 = "9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86",
        strip_prefix = "sentencepiece-0.2.0",
    )

    use_repo_rule(
        http_archive,
        name = "tensorflow",
        url = "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.18.1.tar.gz",
        sha256 = "467c512b631e72ad5c9d5c16b23669bcf89675de630cfbb58f9dde746d34afa8",
        strip_prefix = "tensorflow-2.18.1",
    )

http_archives_extension = module_extension(implementation = http_archives)
