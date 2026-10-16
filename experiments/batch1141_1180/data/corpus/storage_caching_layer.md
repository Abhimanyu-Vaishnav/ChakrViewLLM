# In-Memory Embedding Cache Architecture

## 1. Caching Specifications
- Cache eviction policy: Least Recently Used (LRU) with composite hash keys.
- Maximum cache capacity: 10000 vector embeddings.
- Cache key generation: SHA-256 hash of model identifier plus text content.
- Stale cache invalidation trigger: Document SHA-256 modification or file deletion.
