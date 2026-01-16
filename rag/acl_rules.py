from pathlib import Path
import yaml

class ACLRule:
    def __init__(self, path_prefix: str, allow_roles: list[str]):
        self.path_prefix = path_prefix.strip("/").replace("\\", "/")
        self.allow_roles = allow_roles

    def matches(self, relative_path: str) -> bool:
        return relative_path.startswith(self.path_prefix)


class ACLRules:
    def __init__(self, rules: list[ACLRule]):
        self.rules = rules

    @classmethod
    def load(cls, path: Path) -> "ACLRules":
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rules = []
        for r in data.get("rules", []):
            rules.append(
                ACLRule(
                    path_prefix=r["path_prefix"],
                    allow_roles=r["allow_roles"],
                )
            )
        return cls(rules)

    def resolve_roles(self, relative_path: str) -> list[str] | None:
        """
        Возвращает список ролей, если правило найдено,
        иначе None (решение примет default_allow)
        """
        rel = relative_path.replace("\\", "/")
        for rule in self.rules:
            if rule.matches(rel):
                return rule.allow_roles
        return None
