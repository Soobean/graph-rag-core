import argparse
import asyncio
import logging
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── 샘플 데이터 정의 ──────────────────────────────────────

SAMPLE_DATA_CYPHER = """
// 장르
CREATE (action:Genre {name: 'Action'})
CREATE (drama:Genre {name: 'Drama'})
CREATE (scifi:Genre {name: 'Sci-Fi'})
CREATE (thriller:Genre {name: 'Thriller'})
CREATE (comedy:Genre {name: 'Comedy'})

// 감독
CREATE (nolan:Director {name: 'Christopher Nolan', birth_year: 1970, nationality: 'British'})
CREATE (bong:Director {name: 'Bong Joon-ho', birth_year: 1969, nationality: 'Korean'})
CREATE (tarantino:Director {name: 'Quentin Tarantino', birth_year: 1963, nationality: 'American'})
CREATE (villeneuve:Director {name: 'Denis Villeneuve', birth_year: 1967, nationality: 'Canadian'})

// 배우
CREATE (dicaprio:Actor {name: 'Leonardo DiCaprio', birth_year: 1974, nationality: 'American'})
CREATE (bale:Actor {name: 'Christian Bale', birth_year: 1974, nationality: 'British'})
CREATE (songkangho:Actor {name: 'Song Kang-ho', birth_year: 1967, nationality: 'Korean'})
CREATE (pitt:Actor {name: 'Brad Pitt', birth_year: 1963, nationality: 'American'})
CREATE (chalamet:Actor {name: 'Timothée Chalamet', birth_year: 1995, nationality: 'American'})
CREATE (murphy:Actor {name: 'Cillian Murphy', birth_year: 1976, nationality: 'Irish'})
CREATE (margot:Actor {name: 'Margot Robbie', birth_year: 1990, nationality: 'Australian'})

// 영화
CREATE (inception:Movie {name: 'Inception', title: 'Inception', year: 2010, rating: 8.8, box_office: 836800000})
CREATE (dark_knight:Movie {name: 'The Dark Knight', title: 'The Dark Knight', year: 2008, rating: 9.0, box_office: 1004600000})
CREATE (interstellar:Movie {name: 'Interstellar', title: 'Interstellar', year: 2014, rating: 8.7, box_office: 701700000})
CREATE (oppenheimer:Movie {name: 'Oppenheimer', title: 'Oppenheimer', year: 2023, rating: 8.5, box_office: 952000000})
CREATE (parasite:Movie {name: 'Parasite', title: 'Parasite', year: 2019, rating: 8.5, box_office: 263000000})
CREATE (memories:Movie {name: 'Memories of Murder', title: 'Memories of Murder', year: 2003, rating: 8.1, box_office: 10600000})
CREATE (pulp_fiction:Movie {name: 'Pulp Fiction', title: 'Pulp Fiction', year: 1994, rating: 8.9, box_office: 213900000})
CREATE (django:Movie {name: 'Django Unchained', title: 'Django Unchained', year: 2012, rating: 8.4, box_office: 425400000})
CREATE (dune:Movie {name: 'Dune', title: 'Dune', year: 2021, rating: 8.0, box_office: 402000000})
CREATE (dune2:Movie {name: 'Dune: Part Two', title: 'Dune: Part Two', year: 2024, rating: 8.5, box_office: 711800000})
CREATE (barbie:Movie {name: 'Barbie', title: 'Barbie', year: 2023, rating: 6.8, box_office: 1441800000})

// 감독 - 영화 관계
CREATE (nolan)-[:DIRECTED]->(inception)
CREATE (nolan)-[:DIRECTED]->(dark_knight)
CREATE (nolan)-[:DIRECTED]->(interstellar)
CREATE (nolan)-[:DIRECTED]->(oppenheimer)
CREATE (bong)-[:DIRECTED]->(parasite)
CREATE (bong)-[:DIRECTED]->(memories)
CREATE (tarantino)-[:DIRECTED]->(pulp_fiction)
CREATE (tarantino)-[:DIRECTED]->(django)
CREATE (villeneuve)-[:DIRECTED]->(dune)
CREATE (villeneuve)-[:DIRECTED]->(dune2)

// 배우 - 영화 관계
CREATE (dicaprio)-[:ACTED_IN {role: 'Cobb'}]->(inception)
CREATE (dicaprio)-[:ACTED_IN {role: 'Calvin Candie'}]->(django)
CREATE (bale)-[:ACTED_IN {role: 'Bruce Wayne'}]->(dark_knight)
CREATE (songkangho)-[:ACTED_IN {role: 'Ki-taek'}]->(parasite)
CREATE (songkangho)-[:ACTED_IN {role: 'Park Doo-man'}]->(memories)
CREATE (pitt)-[:ACTED_IN {role: 'Vincent Vega Partner'}]->(pulp_fiction)
CREATE (chalamet)-[:ACTED_IN {role: 'Paul Atreides'}]->(dune)
CREATE (chalamet)-[:ACTED_IN {role: 'Paul Atreides'}]->(dune2)
CREATE (murphy)-[:ACTED_IN {role: 'J. Robert Oppenheimer'}]->(oppenheimer)
CREATE (margot)-[:ACTED_IN {role: 'Barbie'}]->(barbie)

// 영화 - 장르 관계
CREATE (inception)-[:IN_GENRE]->(action)
CREATE (inception)-[:IN_GENRE]->(scifi)
CREATE (dark_knight)-[:IN_GENRE]->(action)
CREATE (dark_knight)-[:IN_GENRE]->(thriller)
CREATE (interstellar)-[:IN_GENRE]->(scifi)
CREATE (interstellar)-[:IN_GENRE]->(drama)
CREATE (oppenheimer)-[:IN_GENRE]->(drama)
CREATE (oppenheimer)-[:IN_GENRE]->(thriller)
CREATE (parasite)-[:IN_GENRE]->(thriller)
CREATE (parasite)-[:IN_GENRE]->(drama)
CREATE (parasite)-[:IN_GENRE]->(comedy)
CREATE (memories)-[:IN_GENRE]->(thriller)
CREATE (memories)-[:IN_GENRE]->(drama)
CREATE (pulp_fiction)-[:IN_GENRE]->(thriller)
CREATE (pulp_fiction)-[:IN_GENRE]->(drama)
CREATE (django)-[:IN_GENRE]->(action)
CREATE (django)-[:IN_GENRE]->(drama)
CREATE (dune)-[:IN_GENRE]->(scifi)
CREATE (dune)-[:IN_GENRE]->(action)
CREATE (dune2)-[:IN_GENRE]->(scifi)
CREATE (dune2)-[:IN_GENRE]->(action)
CREATE (barbie)-[:IN_GENRE]->(comedy)

// 배우 간 협업 관계
CREATE (dicaprio)-[:COLLABORATED_WITH {movie: 'Django Unchained'}]->(pitt)
"""

INDEXES_CYPHER = [
    "CREATE INDEX movie_name IF NOT EXISTS FOR (m:Movie) ON (m.name)",
    "CREATE INDEX actor_name IF NOT EXISTS FOR (a:Actor) ON (a.name)",
    "CREATE INDEX director_name IF NOT EXISTS FOR (d:Director) ON (d.name)",
    "CREATE INDEX genre_name IF NOT EXISTS FOR (g:Genre) ON (g.name)",
]


async def load_sample_data(clear: bool = False) -> None:
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    try:
        async with driver.session(database=settings.neo4j_database) as session:
            # 연결 확인
            result = await session.run("RETURN 1 AS ok")
            await result.consume()
            logger.info("Neo4j 연결 성공")

            if clear:
                logger.info("기존 데이터 삭제 중...")
                await session.run("MATCH (n) DETACH DELETE n")
                logger.info("기존 데이터 삭제 완료")

            # 기존 데이터 확인
            result = await session.run("MATCH (n) RETURN count(n) AS count")
            record = await result.single()
            existing_count = record["count"] if record else 0

            if existing_count > 0 and not clear:
                logger.warning(
                    f"이미 {existing_count}개의 노드가 존재합니다. "
                    "--clear 옵션으로 초기화하거나 기존 데이터와 병합됩니다."
                )

            # 샘플 데이터 로드
            logger.info("샘플 데이터 로드 중...")
            await session.run(SAMPLE_DATA_CYPHER)
            logger.info("샘플 데이터 로드 완료")

            # 인덱스 생성
            for idx_query in INDEXES_CYPHER:
                await session.run(idx_query)
            logger.info(f"인덱스 {len(INDEXES_CYPHER)}개 생성 완료")

            # 결과 확인
            result = await session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label"
            )
            records = [r async for r in result]
            logger.info("── 로드 결과 ──")
            total = 0
            for r in records:
                count = r["count"]
                total += count
                logger.info(f"  {r['label']}: {count}개")

            result = await session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY type"
            )
            rel_records = [r async for r in result]
            rel_total = 0
            for r in rel_records:
                count = r["count"]
                rel_total += count
                logger.info(f"  [{r['type']}]: {count}개")

            logger.info(f"총 노드: {total}개, 총 관계: {rel_total}개")

    finally:
        await driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="샘플 데이터 로드")
    parser.add_argument("--clear", action="store_true", help="기존 데이터 삭제 후 로드")
    args = parser.parse_args()
    asyncio.run(load_sample_data(clear=args.clear))
