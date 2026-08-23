# Reads spaces/isa_features.def and spaces/isa_tiers.def and turns them into CMake variables, so the
# manifests are the single place that says what a tier is. Nothing here decides policy: every value
# comes from a manifest row, and a row that names something the other manifest does not define is a
# configure error rather than a silently skipped tier.
#
#   vecsim_load_manifests(<features.def> <tiers.def>)
#     VECSIM_FEATURE_TOKENS                  all tokens
#     VECSIM_FEATURE_<TOK>_{ARCH,FRAGMENT,LEVEL,FIELD}
#     VECSIM_TIERS                           all tiers, in manifest order
#     VECSIM_TIER_<T>_{ARCH,STEM,PRIORITY,FLAG_TOKENS,GUARANTEE_TOKENS}
#
#   vecsim_tier_flags(<out_var> <tier>)      the compile flag string derived from FLAG_TOKENS alone
#
# The flag string is derived; the runtime predicate is not. GUARANTEE_TOKENS is carried through
# untouched for the C++ side to fold into TierInfo<T>::supported(), because a flag fragment expands
# to a compiler-defined bundle that differs between compilers for the same -march string. Deriving a
# predicate from flags would reintroduce exactly that unsoundness.

# ARM -march levels, low to high. A tier's level is the highest among its tokens, because ARM -march
# strings are monolithic: one level plus "+feature" suffixes, not a union of independent options.
set(VECSIM_ARM_LEVELS "armv8-a;armv8.1-a;armv8.2-a;armv8.3-a;armv8.4-a;armv8.5-a;armv8.6-a;armv9-a")

function(_vecsim_arm_level_rank out_var level)
	list(FIND VECSIM_ARM_LEVELS "${level}" _rank)
	if(_rank EQUAL -1)
		message(FATAL_ERROR
			"vecsim manifest: unknown ARM architecture level '${level}'. Add it to "
			"VECSIM_ARM_LEVELS in cmake/vecsim_manifest.cmake, in ascending order, so the "
			"highest-level comparison stays meaningful.")
	endif()
	set(${out_var} ${_rank} PARENT_SCOPE)
endfunction()

function(vecsim_load_manifests features_def tiers_def)
	if(NOT EXISTS "${features_def}")
		message(FATAL_ERROR "vecsim manifest: cannot read ${features_def}")
	endif()
	if(NOT EXISTS "${tiers_def}")
		message(FATAL_ERROR "vecsim manifest: cannot read ${tiers_def}")
	endif()

	set(_tokens "")
	file(STRINGS "${features_def}" _lines)
	foreach(_line IN LISTS _lines)
		if(_line MATCHES "^FEATURE\\( *([A-Za-z0-9_]+) *, *([A-Z0-9]+) *, *\"([^\"]*)\" *, *([^ ,]+) *, *([A-Za-z0-9_]+) *\\)")
			set(_tok "${CMAKE_MATCH_1}")
			list(APPEND _tokens "${_tok}")
			set(VECSIM_FEATURE_${_tok}_ARCH     "${CMAKE_MATCH_2}" PARENT_SCOPE)
			set(VECSIM_FEATURE_${_tok}_FRAGMENT "${CMAKE_MATCH_3}" PARENT_SCOPE)
			set(VECSIM_FEATURE_${_tok}_LEVEL    "${CMAKE_MATCH_4}" PARENT_SCOPE)
			set(VECSIM_FEATURE_${_tok}_FIELD    "${CMAKE_MATCH_5}" PARENT_SCOPE)
			# also visible inside this function, for the validation below
			set(VECSIM_FEATURE_${_tok}_ARCH     "${CMAKE_MATCH_2}")
			set(VECSIM_FEATURE_${_tok}_FRAGMENT "${CMAKE_MATCH_3}")
			set(VECSIM_FEATURE_${_tok}_LEVEL    "${CMAKE_MATCH_4}")
		endif()
	endforeach()
	if(NOT _tokens)
		message(FATAL_ERROR "vecsim manifest: ${features_def} defined no FEATURE rows. A parser that "
			"silently reads zero rows would disable every tier while the configure still succeeded.")
	endif()

	set(_tiers "")
	file(STRINGS "${tiers_def}" _lines)
	foreach(_line IN LISTS _lines)
		if(_line MATCHES "^TIER\\( *([A-Za-z0-9_]+) *, *([A-Z0-9]+) *, *([A-Za-z0-9_.]+) *, *([0-9]+) *, *\\(([^)]*)\\) *, *\\(([^)]*)\\) *\\)")
			set(_tier "${CMAKE_MATCH_1}")
			set(_arch "${CMAKE_MATCH_2}")
			# The manifest separates tokens with commas for readability; CMake lists are
			# semicolon-separated, so convert rather than relying on a comma string behaving
			# like a list (it does not, and foreach(IN LISTS) silently sees one element).
			string(REPLACE " " "" _flag_tokens "${CMAKE_MATCH_5}")
			string(REPLACE "," ";" _flag_tokens "${_flag_tokens}")
			string(REPLACE " " "" _guar_tokens "${CMAKE_MATCH_6}")
			string(REPLACE "," ";" _guar_tokens "${_guar_tokens}")
			list(APPEND _tiers "${_tier}")

			# Validate before exporting, so a typo is a configure error and not a missing tier.
			foreach(_t IN LISTS _flag_tokens _guar_tokens)
				list(FIND _tokens "${_t}" _known)
				if(_known EQUAL -1)
					message(FATAL_ERROR
						"vecsim manifest: tier ${_tier} names feature token '${_t}', which has no "
						"FEATURE row in ${features_def}.")
				endif()
				if(NOT "${VECSIM_FEATURE_${_t}_ARCH}" STREQUAL "${_arch}")
					message(FATAL_ERROR
						"vecsim manifest: tier ${_tier} is ${_arch} but names token '${_t}', which "
						"is ${VECSIM_FEATURE_${_t}_ARCH}. A tier may only name tokens of its own "
						"architecture.")
				endif()
			endforeach()

			set(VECSIM_TIER_${_tier}_ARCH             "${_arch}"          PARENT_SCOPE)
			set(VECSIM_TIER_${_tier}_STEM             "${CMAKE_MATCH_3}"  PARENT_SCOPE)
			set(VECSIM_TIER_${_tier}_PRIORITY         "${CMAKE_MATCH_4}"  PARENT_SCOPE)
			set(VECSIM_TIER_${_tier}_FLAG_TOKENS      "${_flag_tokens}"   PARENT_SCOPE)
			set(VECSIM_TIER_${_tier}_GUARANTEE_TOKENS "${_guar_tokens}"   PARENT_SCOPE)
		endif()
	endforeach()
	if(NOT _tiers)
		message(FATAL_ERROR "vecsim manifest: ${tiers_def} defined no TIER rows.")
	endif()

	set(VECSIM_FEATURE_TOKENS "${_tokens}" PARENT_SCOPE)
	set(VECSIM_TIERS          "${_tiers}"  PARENT_SCOPE)
endfunction()

# Derive a tier's compile flags from its FLAG_TOKENS. x86 unions the -m fragments in token order;
# ARM emits one -march at the highest level in the token list, then each distinct fragment.
function(vecsim_tier_flags out_var tier)
	set(_arch "${VECSIM_TIER_${tier}_ARCH}")
	set(_tokens "${VECSIM_TIER_${tier}_FLAG_TOKENS}")

	if("${_arch}" STREQUAL "X86")
		set(_parts "")
		foreach(_t IN LISTS _tokens)
			set(_frag "${VECSIM_FEATURE_${_t}_FRAGMENT}")
			list(FIND _parts "${_frag}" _dup)
			if(_frag AND _dup EQUAL -1)
				list(APPEND _parts "${_frag}")
			endif()
		endforeach()
		string(JOIN " " _flags ${_parts})
	elseif("${_arch}" STREQUAL "ARM")
		set(_best_level "armv8-a")
		_vecsim_arm_level_rank(_best_rank "${_best_level}")
		foreach(_t IN LISTS _tokens)
			set(_level "${VECSIM_FEATURE_${_t}_LEVEL}")
			if(NOT "${_level}" STREQUAL "-")
				_vecsim_arm_level_rank(_rank "${_level}")
				if(_rank GREATER _best_rank)
					set(_best_rank ${_rank})
					set(_best_level "${_level}")
				endif()
			endif()
		endforeach()
		set(_suffix "")
		set(_seen "")
		foreach(_t IN LISTS _tokens)
			set(_frag "${VECSIM_FEATURE_${_t}_FRAGMENT}")
			list(FIND _seen "${_frag}" _dup)
			if(_frag AND _dup EQUAL -1)
				list(APPEND _seen "${_frag}")
				string(APPEND _suffix "${_frag}")
			endif()
		endforeach()
		set(_flags "-march=${_best_level}${_suffix}")
	else()
		message(FATAL_ERROR "vecsim manifest: tier ${tier} has unknown architecture '${_arch}'")
	endif()

	set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()
