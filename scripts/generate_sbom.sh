#!/bin/bash

# Software Bill of Materials (SBOM) generation script
# Generates SPDX and CycloneDX format SBOMs

set -e

PROJECT_NAME="bioneuro-olfactory-fusion"
VERSION=$(python -c "import bioneuro_olfactory; print(bioneuro_olfactory.__version__)")
OUTPUT_DIR="sbom"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "🔍 Generating SBOM for ${PROJECT_NAME} v${VERSION}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Generate CycloneDX SBOM (JSON format)
echo "📋 Generating CycloneDX SBOM..."
cyclonedx-py -o "${OUTPUT_DIR}/${PROJECT_NAME}-${VERSION}-cyclonedx.json" \
    --format json \
    --schema-version 1.5 \
    --include-dev

# Generate CycloneDX SBOM (XML format)
echo "📋 Generating CycloneDX XML SBOM..."
cyclonedx-py -o "${OUTPUT_DIR}/${PROJECT_NAME}-${VERSION}-cyclonedx.xml" \
    --format xml \
    --schema-version 1.5 \
    --include-dev

# Generate SPDX SBOM using pip-licenses
echo "📋 Generating SPDX-style license report..."
pip-licenses --format json --output-file "${OUTPUT_DIR}/${PROJECT_NAME}-${VERSION}-licenses.json"

# Generate security vulnerability report
echo "🔒 Running security scan..."
safety check --json --output "${OUTPUT_DIR}/${PROJECT_NAME}-${VERSION}-vulnerabilities.json" || true

# Generate summary report
cat > "${OUTPUT_DIR}/sbom-summary.md" << EOF
# SBOM Generation Report

**Project**: ${PROJECT_NAME}
**Version**: ${VERSION}
**Generated**: ${TIMESTAMP}

## Files Generated

- \`${PROJECT_NAME}-${VERSION}-cyclonedx.json\` - CycloneDX JSON format
- \`${PROJECT_NAME}-${VERSION}-cyclonedx.xml\` - CycloneDX XML format  
- \`${PROJECT_NAME}-${VERSION}-licenses.json\` - License information
- \`${PROJECT_NAME}-${VERSION}-vulnerabilities.json\` - Security scan results

## Usage

Upload these files to your security scanning platform or include them in release artifacts for supply chain transparency.

### Validation

To validate the CycloneDX SBOM:
\`\`\`bash
cyclonedx validate --input-file ${OUTPUT_DIR}/${PROJECT_NAME}-${VERSION}-cyclonedx.json
\`\`\`

### Integration

For CI/CD integration, run this script during your build process and store the SBOM files as build artifacts.
EOF

echo "✅ SBOM generation complete. Files saved to: ${OUTPUT_DIR}/"
echo "📄 Summary report: ${OUTPUT_DIR}/sbom-summary.md"