
BUILT_SOURCES = Doxyfile

DOXYFLAGS = \
	-Dsrcdir=$(srcdir) \
	-Dbuilddir=$(builddir) \
	-Dtop_builddir=$(top_builddir) \
	-Dtop_srcdir=$(top_srcdir) \
	-Dpackage="\"$(PACKAGE)\"" \
	-Dpackage_name="\"$(PACKAGE_NAME)\"" \
	-Dpackage_version="\"$(PACKAGE_VERSION)\""

Doxyfile: Doxyfile.m4
	m4 $(DOXYFLAGS) $< >$@

