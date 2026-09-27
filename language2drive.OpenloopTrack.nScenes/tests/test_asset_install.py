import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.download_assets import install_archive, safe_relative, verify_installed


class AssetInstallTest(unittest.TestCase):
    def fixture(self,root,name='assets/example.txt',payload=b'fixture'):
        archive=root/'bundle.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr(name,payload)
        manifest={'size_bytes':archive.stat().st_size,
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'files':[{'path':name,'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
        return archive,manifest

    def test_verified_install_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,manifest=self.fixture(root);destination=root/'install'
            install_archive(archive,manifest,destination)
            install_archive(archive,manifest,destination)
            self.assertEqual(verify_installed(manifest,destination),[])

    def test_tampering_and_existing_different_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,manifest=self.fixture(root);destination=root/'install'
            install_archive(archive,manifest,destination)
            (destination/'assets/example.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Existing different file'):
                install_archive(archive,manifest,destination)
            archive.write_bytes(archive.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
                install_archive(archive,manifest,destination)

    def test_traversal_and_symlinks_rejected(self):
        for name in ('../outside','/absolute','a/../b','a\\b'):
            with self.assertRaises(ValueError):safe_relative(name)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,manifest=self.fixture(root);destination=root/'install'
            destination.mkdir();(destination/'assets').symlink_to(root,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Symlink'):
                install_archive(archive,manifest,destination)

    def test_manifest_checksum_checked_before_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,manifest=self.fixture(root);destination=root/'install'
            manifest['files'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'Entry SHA-256 mismatch'):
                install_archive(archive,manifest,destination)
            self.assertFalse((destination/'assets/example.txt').exists())

class RedirectAuthenticationTest(unittest.TestCase):
    def test_cross_host_redirect_strips_token(self):
        from scripts.download_assets import AssetRedirectHandler
        from urllib.request import Request
        request=Request('https://hf-mirror.com/datasets/owner/repo/resolve/main/a.zip',
                        headers={'Authorization':'Bearer synthetic-read-token'})
        redirected=AssetRedirectHandler().redirect_request(request,None,302,'Found',{},
                        'https://cdn.example.org/a.zip?signature=synthetic')
        self.assertIsNone(redirected.get_header('Authorization'))
        same=AssetRedirectHandler().redirect_request(request,None,302,'Found',{},
                        'https://hf-mirror.com/signed/a.zip')
        self.assertEqual(same.get_header('Authorization'),'Bearer synthetic-read-token')

    def test_http_redirect_rejected(self):
        from scripts.download_assets import AssetRedirectHandler
        from urllib.request import Request
        with self.assertRaises(ValueError):
            AssetRedirectHandler().redirect_request(Request('https://hf-mirror.com/a'),None,302,'Found',{},'http://example.org/a')
