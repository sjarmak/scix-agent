# Plan: mcp-paper-tools

## 1. Add functions to search.py

### read_paper_section(conn, bibcode, section="full", char_offset=0, limit=5000) -> SearchResult

- Query: `SELECT body, abstract, title FROM papers WHERE bibcode = %s`
- If body exists: use `parse_sections()`, find matching section, slice with char_offset/limit
- If no body: return abstract text with has_body=False
- Return SearchResult with papers=[{section_text, section_name, has_body, char_offset, total_chars, bibcode}]

### search_within_paper(conn, bibcode, query) -> SearchResult

- Query using `ts_headline(body, plainto_tsquery('english', query), 'MaxWords=60,MinWords=20,MaxFragments=5')`
- Also check if body IS NOT NULL
- Return SearchResult with papers=[{bibcode, headline, has_body}]

## 2. Register tools in mcp_server.py

- Add two Tool definitions in list_tools()
- Add two elif branches in \_dispatch_tool()
- Add timeout entries

## 3. Write tests in tests/test_mcp_paper_tools.py

- Test read_paper_section with body (section found, full, pagination)
- Test read_paper_section fallback to abstract
- Test search_within_paper with match and no match
- Test MCP dispatch for both tools
