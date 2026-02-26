"""
Graph Utilities

그래프 시각화 관련 공통 유틸리티 함수 및 상수
"""

from typing import Any

from src.api.schemas.visualization import NodeStyle

# 노드 스타일 — 도메인별로 커스터마이징
# 프로젝트에서 사용하는 노드 라벨에 맞게 수정하세요
NODE_STYLES: dict[str, dict[str, str | float]] = {
    "default": {"color": "#BDBDBD", "icon": "circle", "size": 1.0},
}


def get_node_style(label: str) -> NodeStyle:
    """노드 라벨에 따른 스타일 반환"""
    style_data = NODE_STYLES.get(label, NODE_STYLES["default"])
    return NodeStyle(
        color=str(style_data["color"]),
        icon=str(style_data["icon"]),
        size=float(style_data["size"]),
    )


def sanitize_props(props: dict[str, Any]) -> dict[str, Any]:
    """Neo4j 속성을 JSON 직렬화 가능한 형태로 변환"""
    sanitized: dict[str, Any] = {}
    for key, value in props.items():
        if "embedding" in key.lower():
            continue
        if hasattr(value, "isoformat"):
            sanitized[key] = value.isoformat()
        elif hasattr(value, "__str__") and not isinstance(
            value, (str, int, float, bool, list, dict, type(None))
        ):
            sanitized[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [
                v.isoformat() if hasattr(v, "isoformat") else v for v in value
            ]
    return sanitized
