# vector-snapshot-iteration (delta)

> On merge, the requirements below fold into
> `openspec/specs/vector-snapshot-iteration/spec.md`.

## ADDED Requirements

### Requirement: Vector batch iteration is consistent as of construction
When snapshot iteration is enabled, a vector batch iterator (KNN, hybrid, and
cursor-paginated vector aggregation) over an HNSW index SHALL return results
consistent with the index state at the time the iterator was constructed.

#### Scenario: Vectors added after construction are not returned
- **WHEN** a batch iterator is constructed over an HNSW index
- **AND** new vectors are added to the index while the iterator is being consumed
- **THEN** results returned by the iterator SHALL NOT include the vectors added
  after construction

#### Scenario: Vectors deleted after construction remain visible
- **WHEN** a batch iterator is constructed over an HNSW index containing a
  vector `v`
- **AND** `v` is deleted while the iterator is being consumed
- **THEN** `v`'s **slot is not recycled** while the snapshot can still read it —
  its id is below the reclaim horizon (`maxVisibleCount`), so the SWAP is deferred
  — and the snapshot continues to read `v`'s own embedding and identity, never a
  different element wrongly occupying `v`'s slot. (Slots at/above the horizon —
  ids no live snapshot can see — still recycle promptly.)

#### Scenario: Deleted-flag consistency is weak (the chosen contract)
> **Decision (task 4.8).** `markDelete` flips the deleted flag **in place** in the
> single-version `idToMetaData` container; copy-on-write versions only the graph,
> and SWAP deferral protects only against *slot recycling*, not against a flag
> flip on the slot's own metadata.
- **WHEN** a snapshot is captured over an index containing a live vector `v`
- **AND** `v` is marked deleted while the snapshot is held
- **THEN** the snapshot MAY observe `v` as deleted (and filter it out), even
  though `v` was live at capture. This is the **accepted weak contract**: "no
  *new* vectors after T, crash-safe, slot-stable." Strict as-of-capture
  deleted-flags would require versioning the metadata flag and is tracked as
  separate work (not built).

#### Scenario: Repeated reads of the same snapshot are stable
- **WHEN** a snapshot iterator is constructed
- **AND** the index is modified concurrently by inserts and deletes
- **THEN** the set of candidate elements the iterator traverses SHALL be the
  point-in-time set captured at construction, unaffected by those modifications

### Requirement: Iteration does not hold the index lock
When snapshot iteration is enabled, a vector batch iterator SHALL acquire the
index read lock only to construct (capture) the snapshot, and SHALL NOT hold any
index lock while producing results.

#### Scenario: Writers progress during a long read
- **WHEN** a long-running vector iteration (e.g. a paginated cursor) is in
  progress
- **AND** a concurrent client inserts or deletes vectors
- **THEN** the inserts and deletes SHALL complete without waiting for the
  iteration to finish

### Requirement: Snapshot memory is bounded
> **Status: NOT yet implemented.** The reclaim horizon keeps compaction flowing
> for above-horizon churn, but nothing yet caps a single long-lived cursor's
> retained memory or its indefinitely-deferred below-horizon slots. This
> requirement is the remaining production-safety gap (tasks.md 6.2).

A configured limit SHALL bound the memory retained by live snapshots. When the
limit would be exceeded, the system SHALL take a deterministic, documented
action rather than growing retained memory without bound.

#### Scenario: Limit is enforced for a long-lived cursor
- **WHEN** snapshot iteration is enabled with a memory/lifetime limit
- **AND** a cursor is held open while concurrent writes accumulate superseded
  graph versions up to the limit
- **THEN** the system SHALL enforce the limit by the configured on-breach action
  (refuse new snapshots or terminate the offending cursor with a clear error)
- **AND** retained snapshot memory SHALL NOT grow beyond the configured bound

### Requirement: No regression when snapshots are inactive
When no snapshot is active, vector indexing and query operations SHALL behave as
before this change, with no additional copy-on-write or locking overhead.

#### Scenario: In-place writes with no active snapshot
- **WHEN** no snapshot iterator is live
- **AND** vectors are inserted or deleted
- **THEN** the index SHALL mutate its graph in place without copy-on-write
