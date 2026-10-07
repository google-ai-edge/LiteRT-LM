load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

DATA_FILES = glob(
    ["espeak-ng-data/**"],
    exclude = [
        "espeak-ng-data/intonations",
        "espeak-ng-data/mbrola_ph",
        "espeak-ng-data/phondata-manifest",
        "espeak-ng-data/voices/!v/Mr serious",
    ],
)

PHSOURCE_FILES = glob(["phsource/**"])

DICTSOURCE_FILES = glob(["dictsource/**"])

cc_library(
    name = "config_header",
    hdrs = ["android/jni/include/config.h"],
    strip_include_prefix = "android/jni/include/",
)

cc_library(
    name = "ucd_header",
    hdrs = ["src/ucd-tools/src/include/ucd/ucd.h"],
    strip_include_prefix = "src/ucd-tools/src/include/",
)

cc_library(
    name = "speech_player_header",
    hdrs = ["src/speechPlayer/include/speechPlayer.h"],
    strip_include_prefix = "src/speechPlayer/include",
)

cc_library(
    name = "espeak_speak_lib_header",
    hdrs = ["src/include/espeak/speak_lib.h"],
)

cc_library(
    name = "espeak_ng_speak_lib_header",
    hdrs = ["src/include/espeak-ng/speak_lib.h"],
    strip_include_prefix = "src/include/",
)

cc_library(
    name = "espeak_ng_base_headers",
    hdrs = [
        "src/include/espeak-ng/encoding.h",
        "src/include/espeak-ng/espeak_ng.h",
    ],
    strip_include_prefix = "src/include/",
    deps = [
        ":espeak_ng_speak_lib_header",
    ],
)

cc_library(
    name = "noinst_headers",
    hdrs = glob([
        "src/libespeak-ng/*.h",
        "src/speechPlayer/src/*.h",
    ]),
    deps = [
        ":espeak_ng_base_headers",
        ":espeak_ng_speak_lib_header",
        ":speech_player_header",
    ],
)

cc_library(
    name = "libespeak_ng_la_sources",
    srcs = glob([
        "src/libespeak-ng/*.c",
        "src/speechPlayer/src/*.cpp",
        "src/ucd-tools/src/*.c",
    ]),
    copts = [
        "-Wno-missing-braces",
        "-Wno-unused-variable",
        "-Wno-unused-result",
        "-Wno-error=unused-result",
        "-Wno-format",
    ],
    defines = [
        "N_PATH_HOME=4096",
        "PATH_ESPEAK_DATA=\\\"/usr/share/espeak-ng-data\\\"",
    ],
    deps = [
        ":config_header",
        ":espeak_ng_base_headers",
        ":espeak_ng_speak_lib_header",
        ":espeak_speak_lib_header",
        ":noinst_headers",
        ":speech_player_header",
        ":ucd_header",
    ],
)

cc_library(
    name = "espeak_ng",
    deps = [
        ":espeak_ng_base_headers",
        ":espeak_ng_speak_lib_header",
        ":libespeak_ng_la_sources",
    ],
)

cc_library(
    name = "espeak_ng_header",
    hdrs = [
        "src/include/espeak-ng/espeak_ng.h",
    ],
    strip_include_prefix = "src/include/",
    deps = [
        ":espeak_ng_speak_lib_header",
    ],
)

cc_binary(
    name = "espeak-ng-bin",
    srcs = ["src/espeak-ng.c"],
    copts = [
        "-Wno-missing-braces",
        "-Wno-unused-variable",
        "-Wno-unused-result",
        "-Wno-error=unused-result",
        "-Wno-format",
    ],
    defines = [
        "N_PATH_HOME=4096",
        "PATH_ESPEAK_DATA=\\\"/usr/share/espeak-ng-data\\\"",
    ],
    data = DATA_FILES + PHSOURCE_FILES,
    deps = [
        ":config_header",
        ":espeak_ng_base_headers",
        ":espeak_ng_header",
        ":espeak_ng_speak_lib_header",
        ":libespeak_ng_la_sources",
    ],
)

genrule(
    name = "intonations",
    srcs = PHSOURCE_FILES + DATA_FILES,
    outs = ["espeak-ng-data/intonations"],
    cmd = """
        WORK_DIR="$(RULEDIR)/.work_intonations"
        rm -rf "$$WORK_DIR"
        mkdir -p "$$WORK_DIR/espeak-ng-data" "$$WORK_DIR/phsource" "$(RULEDIR)/espeak-ng-data"
        cp -R external/espeak_ng/espeak-ng-data/* "$$WORK_DIR/espeak-ng-data/" 2>/dev/null || true
        cp -R external/espeak_ng/phsource/* "$$WORK_DIR/phsource/" 2>/dev/null || true
        chmod -R 755 "$$WORK_DIR"
        $(location :espeak-ng-bin) --path="$$WORK_DIR" --compile-intonations
        cp -f "$$WORK_DIR/espeak-ng-data/intonations" "$@"
        rm -rf "$$WORK_DIR"
    """,
    tools = [":espeak-ng-bin"],
)

genrule(
    name = "phonemes",
    srcs = [
        ":intonations",
    ] + PHSOURCE_FILES + DATA_FILES,
    outs = [
        "espeak-ng-data/phondata",
        "espeak-ng-data/phondata-manifest",
        "espeak-ng-data/phonindex",
        "espeak-ng-data/phontab",
    ],
    cmd = """
        WORK_DIR="$(RULEDIR)/.work_phonemes"
        rm -rf "$$WORK_DIR"
        mkdir -p "$$WORK_DIR/espeak-ng-data" "$$WORK_DIR/phsource" "$(RULEDIR)/espeak-ng-data"
        cp -R external/espeak_ng/espeak-ng-data/* "$$WORK_DIR/espeak-ng-data/" 2>/dev/null || true
        cp -R external/espeak_ng/phsource/* "$$WORK_DIR/phsource/" 2>/dev/null || true
        cp -f $(location :intonations) "$$WORK_DIR/espeak-ng-data/intonations"
        chmod -R 755 "$$WORK_DIR"
        $(location :espeak-ng-bin) --path="$$WORK_DIR" --compile-phonemes
        for f in phondata phondata-manifest phonindex phontab; do
          cp -f "$$WORK_DIR/espeak-ng-data/$$f" "$(RULEDIR)/espeak-ng-data/$$f"
        done
        rm -rf "$$WORK_DIR"
    """,
    tools = [":espeak-ng-bin"],
)

LOCALES = [
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "pt",
    "zh",
]

[genrule(
    name = "%s_dictionary" % locale,
    srcs = [
        ":phonemes",
        ":intonations",
    ] + DICTSOURCE_FILES + DATA_FILES + PHSOURCE_FILES,
    outs = [
        "espeak-ng-data/%s_dict" % locale,
    ] if locale != "zh" else [
        "espeak-ng-data/cmn_dict",
    ],
    cmd = """
        WORK_DIR="$(RULEDIR)/.work_{locale}"
        rm -rf "$$WORK_DIR"
        mkdir -p "$$WORK_DIR/espeak-ng-data" "$$WORK_DIR/phsource" "$$WORK_DIR/dictsource" "$(RULEDIR)/espeak-ng-data"
        cp -R external/espeak_ng/espeak-ng-data/* "$$WORK_DIR/espeak-ng-data/" 2>/dev/null || true
        cp -R external/espeak_ng/phsource/* "$$WORK_DIR/phsource/" 2>/dev/null || true
        cp -R external/espeak_ng/dictsource/* "$$WORK_DIR/dictsource/" 2>/dev/null || true
        for f in $(locations :phonemes) $(location :intonations); do
          cp -f "$$f" "$$WORK_DIR/espeak-ng-data/$$(basename $$f)"
        done
        chmod -R 755 "$$WORK_DIR"
        EXECROOT=$$(pwd)
        (cd "$$WORK_DIR/dictsource" && "$$EXECROOT/$(location :espeak-ng-bin)" --path="$$EXECROOT/$$WORK_DIR" --compile={locale})
        cp -f "$$WORK_DIR/espeak-ng-data/$$(basename $@)" "$@"
        rm -rf "$$WORK_DIR"
    """.format(locale = locale),
    tools = [":espeak-ng-bin"],
) for locale in LOCALES]

filegroup(
    name = "espeak_ng_data",
    srcs = DATA_FILES + [
        ":intonations",
        ":phonemes",
    ] + [":%s_dictionary" % locale for locale in LOCALES],
)
