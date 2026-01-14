/**
 * Wyscout URL Capture v2 - Catches navigation links
 * Paste in DevTools Console (F12) on wyscout.hudl.com
 */
(()=>{
    window.W = window.W || [];

    const isVideo = u => u && typeof u === 'string' &&
        (u.includes('cdn') && u.includes('wyscout') && u.includes('.mp4'));

    const capture = (url, src) => {
        if (W.includes(url)) return;
        W.push(url);
        console.log(`%c✓ [${src}] #${W.length}: ${url.split('?')[0].split('/').pop()}`, 'color:#0f0; font-weight:bold');
    };

    // Intercept ALL link clicks before navigation
    document.addEventListener('click', e => {
        // Check clicked element and all parents for links
        let el = e.target;
        while (el && el !== document) {
            if (el.tagName === 'A' && el.href && isVideo(el.href)) {
                capture(el.href, 'link-click');
                // Don't prevent - let user download naturally, we just capture URL
            }
            // Also check for buttons that might trigger downloads
            if (el.tagName === 'BUTTON' || el.role === 'button') {
                // Check nearby links or data attributes
                const nearbyLink = el.querySelector('a[href*="cdn"]') ||
                                   el.closest('[data-url]') ||
                                   document.querySelector('a[href*="cdn5download"]');
                if (nearbyLink) {
                    const url = nearbyLink.href || nearbyLink.dataset.url;
                    if (isVideo(url)) capture(url, 'button');
                }
            }
            el = el.parentElement;
        }
    }, true);

    // Watch for dynamically added links
    const observer = new MutationObserver(mutations => {
        mutations.forEach(m => {
            m.addedNodes.forEach(node => {
                if (node.nodeType === 1) {
                    // Check the node itself
                    if (node.tagName === 'A' && isVideo(node.href)) {
                        capture(node.href, 'dom-added');
                    }
                    // Check children
                    node.querySelectorAll?.('a[href*="cdn5download"]').forEach(a => {
                        if (isVideo(a.href)) capture(a.href, 'dom-child');
                    });
                }
            });
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});

    // Intercept window.open (popups)
    const origOpen = window.open;
    window.open = function(url, ...args) {
        if (isVideo(url)) capture(url, 'window.open');
        return origOpen.apply(this, [url, ...args]);
    };

    // Intercept location changes
    const origAssign = location.assign;
    location.assign = function(url) {
        if (isVideo(url)) capture(url, 'location.assign');
        return origAssign.call(this, url);
    };

    // Watch for href attribute changes
    const origSetAttribute = Element.prototype.setAttribute;
    Element.prototype.setAttribute = function(name, value) {
        if (name === 'href' && isVideo(value)) capture(value, 'setAttribute');
        return origSetAttribute.call(this, name, value);
    };

    // Periodically scan page for download links
    setInterval(() => {
        document.querySelectorAll('a[href*="cdn5download.wyscout"]').forEach(a => {
            if (isVideo(a.href)) capture(a.href, 'scan');
        });
    }, 1000);

    // Helper functions
    window.urls = () => {
        console.log('\n=== CAPTURED URLS ===');
        W.forEach((url, i) => console.log(`${i+1}. ${url}`));
        navigator.clipboard.writeText(W.join('\n')).then(() =>
            console.log('%c✓ Copied to clipboard!', 'color:#0f0'));
        return W;
    };

    window.save = () => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([W.join('\n')], {type:'text/plain'}));
        a.download = 'wyscout_urls.txt';
        a.click();
        console.log('%c✓ Saved to wyscout_urls.txt', 'color:#0f0');
    };

    window.clear = () => { W.length = 0; console.log('Cleared'); };

    // Manual add
    window.add = (url) => {
        if (url && !W.includes(url)) {
            W.push(url);
            console.log(`%c✓ Manually added #${W.length}`, 'color:#0f0');
        }
    };

    console.log(`%c
╔════════════════════════════════════════════════════════════╗
║  WYSCOUT CAPTURE v2 - Navigation Link Detection            ║
╠════════════════════════════════════════════════════════════╣
║  Now watching for download links...                        ║
║                                                            ║
║  Commands:                                                 ║
║    urls()     - Show & copy all URLs                       ║
║    save()     - Download as urls.txt                       ║
║    add(url)   - Manually add a URL                         ║
║    clear()    - Clear all URLs                             ║
║                                                            ║
║  Captured: ${W.length} URLs                                         ║
╚════════════════════════════════════════════════════════════╝
`, 'color:#0ff; font-weight:bold');

})();
