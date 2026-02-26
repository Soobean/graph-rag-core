"""Response Metadata Builder — 파이프라인 결과를 API 응답 메타데이터로 변환"""

from typing import Any


def build_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """스트리밍용 메타데이터 구성 (graph_data 포함)"""
    metadata: dict[str, Any] = {
        "intent": state.get("intent", "unknown"),
        "intent_confidence": state.get("intent_confidence", 0.0),
        "entities": state.get("entities", {}),
        "cypher_query": state.get("cypher_query", ""),
        "result_count": state.get("result_count", 0),
        "execution_path": state.get("execution_path", []),
    }

    graph_data = build_graph_data(state)
    if graph_data:
        metadata["graph_data"] = graph_data
    else:
        tabular_data = build_tabular_data(state)
        if tabular_data:
            metadata["tabular_data"] = tabular_data

    return metadata


def build_graph_data(state: dict[str, Any], limit: int = 200) -> dict[str, Any] | None:
    """스트리밍용 그래프 데이터 구축"""
    graph_results = state.get("graph_results", [])
    if not graph_results:
        return None

    # resolved_entities 이름 집합 (쿼리 시작점 판별용)
    resolved_names: set[str] = set()
    for entity in state.get("resolved_entities", []):
        name = entity.get("name", entity.get("original_value", ""))
        if name:
            resolved_names.add(name)

    # entities 이름 집합 (fallback)
    entity_names: set[str] = set()
    for values in state.get("entities", {}).values():
        entity_names.update(values)

    start_names = resolved_names or entity_names
    default_style = {"color": "#607D8B", "icon": "circle", "size": 30}

    nodes_map: dict[str, dict[str, Any]] = {}
    edges_list: list[dict[str, Any]] = []
    query_entity_ids: list[str] = []
    result_entity_ids: list[str] = []

    for row in graph_results[:limit]:
        for value in row.values():
            if not isinstance(value, dict):
                continue

            # 노드 감지
            if "labels" in value and isinstance(value.get("labels"), list):
                elem_id = value.get("elementId", f"node_{len(nodes_map)}")
                node_id = str(value.get("id", elem_id))
                if node_id in nodes_map:
                    continue
                labels = value.get("labels", [])
                props = value.get("properties", {}) if isinstance(value.get("properties"), dict) else {}
                name = props.get("name", "")
                label = labels[0] if labels else "Node"

                if name in start_names:
                    role, depth = "start", 0
                    query_entity_ids.append(node_id)
                else:
                    role, depth = "end", 2
                    result_entity_ids.append(node_id)

                nodes_map[node_id] = {
                    "id": node_id,
                    "label": label,
                    "name": name or "Unknown",
                    "properties": {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))},
                    "group": label,
                    "role": role,
                    "depth": depth,
                    "style": default_style,
                }

            # 관계 감지 (labels가 없는 dict만)
            elif "labels" not in value and "type" in value and ("startNodeId" in value or "start" in value):
                edge_id = str(value.get("id", value.get("elementId", f"edge_{len(edges_list)}")))
                source = str(value.get("startNodeId", value.get("start", "")))
                target = str(value.get("endNodeId", value.get("end", "")))
                if edge_id and source and target:
                    edges_list.append({
                        "id": edge_id,
                        "source": source,
                        "target": target,
                        "label": value.get("type", "RELATED"),
                        "properties": value.get("properties", {}),
                    })

    valid_edges = [e for e in edges_list if e["source"] in nodes_map and e["target"] in nodes_map]

    if not nodes_map:
        return None

    return {
        "nodes": list(nodes_map.values())[:limit],
        "edges": valid_edges,
        "node_count": min(len(nodes_map), limit),
        "edge_count": len(valid_edges),
        "query_entity_ids": query_entity_ids,
        "result_entity_ids": result_entity_ids,
        "has_more": len(graph_results) > limit,
    }


def build_tabular_data(state: dict[str, Any], limit: int = 100) -> dict[str, Any] | None:
    """집계 쿼리 결과를 테이블 형식으로 변환"""
    graph_results = state.get("graph_results", [])
    if not graph_results:
        return None

    first_row = graph_results[0]
    columns = [k for k, v in first_row.items() if isinstance(v, (str, int, float, bool, type(None)))]
    if not columns:
        return None

    rows = [{col: row.get(col) for col in columns} for row in graph_results[:limit]]
    return {
        "columns": columns,
        "rows": rows,
        "total_count": len(graph_results),
        "has_more": len(graph_results) > limit,
    }
