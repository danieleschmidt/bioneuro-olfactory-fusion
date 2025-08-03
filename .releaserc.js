module.exports = {
  branches: [
    'main',
    {
      name: 'develop',
      prerelease: true
    },
    {
      name: 'feature/*',
      prerelease: true
    }
  ],
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md',
        changelogTitle: '# Changelog\n\nAll notable changes to the BioNeuro-Olfactory-Fusion project will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).'
      }
    ],
    [
      '@semantic-release/exec',
      {
        prepareCmd: 'python scripts/update_version.py ${nextRelease.version}',
        publishCmd: 'echo "Version ${nextRelease.version} released"'
      }
    ],
    [
      '@semantic-release/github',
      {
        assets: [
          { path: 'dist/*.whl', label: 'Python Wheel' },
          { path: 'dist/*.tar.gz', label: 'Source Distribution' },
          { path: 'docs/build/html.zip', label: 'Documentation' }
        ]
      }
    ],
    [
      '@semantic-release/git',
      {
        assets: ['CHANGELOG.md', 'bioneuro_olfactory/__init__.py', 'pyproject.toml'],
        message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
      }
    ]
  ]
};