#!/bin/bash
CLANG_FMT_SRCS=$(find ./src/ \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' \))
CLANG_FMT_TESTS="$(find ./tests/ -type d \( -path ./tests/unit/build \) -prune -false -o  \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' \))"

E=0
for filename in $CLANG_FMT_SRCS $CLANG_FMT_TESTS; do
    if [[ $FIX == 1 ]]; then
        clang-format --verbose -style=file -i $filename
    else
        echo "Checking $filename ..."
        { clang-format -style=file -Werror --dry-run $filename; (( E |= $? )); } || true
    fi
done
exit $E
