import logging

log = logging.getLogger(__name__)


class ACLRuntimeFilter:
    def __init__(self, cfg):
        self.enabled = cfg.acl
        self.default_allow = cfg.default_allow

    def filter(self, results: list[dict], user_role: str) -> list[dict]:
        if not self.enabled:
            return results

        filtered = []

        for r in results:
            chunk = r.get("main_chunk")
            if not chunk:
                log.debug("Skipping result with no main_chunk.")
                continue

            roles = getattr(chunk, "allowed_roles", "")

            if self._allowed(roles, user_role):
                log.debug(
                    "ACL check PASSED for chunk %s. Roles: '%s', User Role: '%s'",
                    chunk.id,
                    roles,
                    user_role,
                )
                filtered.append(r)
            else:
                log.debug(
                    "ACL check DENIED for chunk %s. Roles: '%s', User Role: '%s'",
                    chunk.id,
                    roles,
                    user_role,
                )

        log.info(
            "ACL runtime filter: before=%d after=%d role=%s",
            len(results), len(filtered), user_role
        )
        return filtered

    def _allowed(self, roles_str: str, user_role: str) -> bool:
        if roles_str == "*":
            return True
        if not roles_str:
            return False

        roles = roles_str.split("|")
        return user_role in roles

