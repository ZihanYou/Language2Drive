"""Private publishing guardrails; tests never contact the network."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    'hub_assets', Path(__file__).parents[1] / 'scripts/hub_assets.py')
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)


def test_disable_proxy_environment(monkeypatch):
    for name in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy'):
        monkeypatch.setenv(name, 'http://do-not-use.invalid')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('NO_PROXY', 'localhost')
    monkeypatch.setenv('no_proxy', 'localhost')
    hub.direct_network()
    assert not any(k.lower() in ('http_proxy', 'https_proxy', 'all_proxy') for k in hub.os.environ)
    assert hub.os.environ['NO_PROXY'] == '*'
    assert 'HF_HUB_OFFLINE' not in hub.os.environ


def test_proxy_requires_opt_in(monkeypatch):
    monkeypatch.setenv('https_proxy', 'http://localhost:1234')
    monkeypatch.setenv('NO_PROXY', 'localhost')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    hub.direct_network(keep_proxy=True)
    assert hub.os.environ['https_proxy'] == 'http://localhost:1234'
    assert hub.os.environ['NO_PROXY'] == 'localhost'
    assert 'HF_HUB_OFFLINE' not in hub.os.environ


def api(private=True):
    mock = Mock()
    mock.whoami.return_value = {'name': 'owner'}
    mock.repo_info.return_value = SimpleNamespace(private=private)
    return mock


def test_repository_and_prefix():
    client = api()
    assert hub.ensure_private_repo(client) == 'owner/Language2Drive'
    client.create_repo.assert_called_once_with('owner/Language2Drive', repo_type='dataset', private=True, exist_ok=True)
    client.update_repo_settings.assert_called_once_with('owner/Language2Drive', repo_type='dataset', private=True)
    assert hub.PREFIX == 'langauge2drive.RealSim'


@pytest.mark.parametrize('target', ['other/Language2Drive', 'owner/Other'])
def test_wrong_repository_has_no_write(target):
    client = api()
    with pytest.raises(ValueError):
        hub.ensure_private_repo(client, target)
    client.create_repo.assert_not_called()


def test_unverified_privacy_fails_closed():
    with pytest.raises(RuntimeError):
        hub.ensure_private_repo(api(private=False))


def test_mirror_is_anonymous_only():
    with pytest.raises(ValueError):
        hub.validate_endpoint(hub.MIRROR_ENDPOINT, 'test-credential')
    hub.validate_endpoint(hub.MIRROR_ENDPOINT, None)
    hub.validate_endpoint(hub.OFFICIAL_ENDPOINT, 'test-credential')


@pytest.mark.parametrize('endpoint', [
    'http://hf-mirror.com', 'https://hf-mirror.com.evil.invalid',
    'https://hf-mirror.com' + '@' + 'evil.invalid', 'https://unknown.invalid'])
def test_arbitrary_hosts_are_rejected(endpoint):
    with pytest.raises(ValueError):
        hub.validate_endpoint(endpoint, 'test-credential')
